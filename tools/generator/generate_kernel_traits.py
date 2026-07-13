#!/usr/bin/env python3
"""Generate kernel instantiation candidates and host fwd trait dispatch.

`kernel_traits.yaml` is the only hand-maintained trait database.  Generated
outputs are deliberately architecture-specific so generating one target cannot
overwrite another target's candidate set.
"""

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("kernel_traits.yaml")
OUT_DIR = Path(__file__).with_name("out")
REGISTRY = ROOT / "csrc/flash_attn/flash_dispatch/generated/fwd_kernel_traits_registry.h"
FWD_REQUIRED = {
    "hdim_qk", "hdim_v", "block_m", "block_n", "k_nwarps",
    "Is_Q_in_regs", "Share_Q_K_smem", "block_num_per_ap",
}


def load_database():
    with SOURCE.open() as source:
        database = yaml.safe_load(source)
    if database.get("version") != 1:
        raise ValueError("unsupported kernel traits schema version")
    return database["architectures"]


def validate(architectures):
    for arch, operations in architectures.items():
        for operation, section in operations.items():
            candidates = section.get("candidates", [])
            ids = set()
            for candidate in candidates:
                if operation in ("fwd", "fwd_split"):
                    missing = FWD_REQUIRED - candidate.keys()
                    if missing:
                        raise ValueError(f"{arch}/{operation}/{candidate.get('id')}: missing {sorted(missing)}")
                    if candidate["id"] in ids:
                        raise ValueError(f"duplicate candidate id: {arch}/{operation}/{candidate['id']}")
                    ids.add(candidate["id"])
            for rule in section.get("dispatch", []):
                if rule["candidate"] not in ids:
                    raise ValueError(f"{arch}/{operation}: dispatch references unknown candidate {rule['candidate']}")


def kernel_candidates(architectures, arch):
    """Keep the legacy generator input shape, excluding host-only metadata."""
    candidates = {}
    for operation, section in architectures[arch].items():
        configs = []
        for candidate in section.get("candidates", []):
            config = dict(candidate)
            config.pop("id", None)
            config.pop("block_num_per_ap", None)
            configs.append(config)
        candidates[operation] = configs
    return candidates


def cpp_bool(value):
    return "true" if value else "false"


def make_meta(arch, operation, candidate):
    common = (
        f"Arch::{arch}, {candidate['hdim_qk']}, "
        f"{candidate['block_m']}, {candidate['block_n']}, {candidate['k_nwarps']}, "
        f"{cpp_bool(candidate['Is_Q_in_regs'])}, {cpp_bool(candidate['Share_Q_K_smem'])}"
    )
    if operation == "fwd":
        return (
            f"make_fwd_meta(Arch::{arch}, {candidate['hdim_qk']}, {candidate['hdim_v']}, "
            f"{candidate['block_m']}, {candidate['block_n']}, {candidate['k_nwarps']}, "
            f"{cpp_bool(candidate['Is_Q_in_regs'])}, {cpp_bool(candidate['Share_Q_K_smem'])}, "
            f"5, 0, {candidate['block_num_per_ap']})"
        )
    return f"make_fwd_split_meta({common}, {candidate['block_num_per_ap']})"


def rule_condition(arch, operation, candidate, rule):
    terms = [f"arch == Arch::{arch}", f"headdim == {candidate['hdim_qk']}"]
    if "dropout" in rule:
        terms.append("is_dropout" if rule["dropout"] else "!is_dropout")
    if "mla" in rule:
        terms.append("is_mla" if rule["mla"] else "!is_mla")
    if "seqlen_q_max" in rule:
        terms.append(f"params.seqlen_q <= {rule['seqlen_q_max']}")
    return " && ".join(terms)


def emit_selector(architectures, operation):
    return_type = "FwdKernelMeta" if operation == "fwd" else "FwdSplitKernelMeta"
    name = "select_fwd_meta" if operation == "fwd" else "select_fwd_split_meta"
    effective = "fwd_effective_headdim" if operation == "fwd" else "fwd_split_effective_headdim"
    lines = [f"inline {return_type} {name}(const Flash_fwd_params &params) {{",
             "    const int arch = params.arch;",
             f"    const int headdim = {effective}(params.d);"]
    if operation == "fwd":
        lines.extend([
            "    const bool is_dropout = params.p_dropout < 1.0f;",
            "    const bool is_mla = headdim == 192 && params.d_value_rounded == 128;",
        ])
    for arch, operations in architectures.items():
        section = operations[operation]
        candidates = {candidate["id"]: candidate for candidate in section["candidates"]}
        for rule in section.get("dispatch", []):
            candidate = candidates[rule["candidate"]]
            condition = rule_condition(arch, operation, candidate, rule)
            lines.append(f"    if ({condition}) {{ return {make_meta(arch, operation, candidate)}; }}")
    lines.extend(["    return {};", "}", ""])
    return lines


def emit_switch_macro(architectures, operation):
    macro = "FWD_META_SWITCH" if operation == "fwd" else "FWD_SPLIT_META_SWITCH"
    args = (
        "META, kArch, kHeadDim, kHeadDimV, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, ..."
        if operation == "fwd" else
        "META, kArch, kHeadDim, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, ..."
    )
    check = "check_fwd_meta_supported" if operation == "fwd" else "check_fwd_split_meta_supported"
    unique = []
    seen = set()
    for arch, operations in architectures.items():
        for candidate in operations[operation]["candidates"]:
            key = (arch, candidate["hdim_qk"], candidate["hdim_v"], candidate["block_m"], candidate["block_n"],
                   candidate["k_nwarps"], candidate["Is_Q_in_regs"], candidate["Share_Q_K_smem"])
            if key not in seen:
                seen.add(key)
                unique.append((arch, candidate))

    lines = [f"#define {macro}({args}) \\",
             "    [&] { \\",
             "        const auto &meta__ = (META); \\",
             f"        mcFlashAttn::{check}(meta__); \\"]
    for arch, candidate in unique:
        gate = f"(kArch) == Arch::{arch} && (kHeadDim) == {candidate['hdim_qk']}"
        if operation == "fwd":
            gate += f" && (kHeadDimV) == {candidate['hdim_v']}"
        predicate = (
            f"meta__.block_m == {candidate['block_m']} && meta__.block_n == {candidate['block_n']} && "
            f"meta__.nwarps == {candidate['k_nwarps']} && "
            f"meta__.is_q_in_regs == {cpp_bool(candidate['Is_Q_in_regs'])} && "
            f"meta__.share_q_k_smem == {cpp_bool(candidate['Share_Q_K_smem'])}"
        )
        lines.extend([
            f"        if constexpr ({gate}) {{ \\",
            f"            if ({predicate}) {{ \\",
            f"                constexpr static int kBlockM = {candidate['block_m']}; \\",
            f"                constexpr static int kBlockN = {candidate['block_n']}; \\",
            f"                constexpr static int kNWarps = {candidate['k_nwarps']}; \\",
            f"                constexpr static bool Is_Q_in_regs = {cpp_bool(candidate['Is_Q_in_regs'])}; \\",
            f"                constexpr static bool Share_Q_K_smem = {cpp_bool(candidate['Share_Q_K_smem'])}; \\",
            "                return __VA_ARGS__(); \\",
            "            } \\",
            "        } \\",
        ])
    lines.extend([
        "        throw std::invalid_argument(\"Unsupported generated fwd kernel tuple\"); \\",
        "    }()",
        "",
    ])
    return lines


def write_registry(architectures):
    lines = [
        "// Generated by tools/generator/generate_kernel_traits.py. Do not edit.",
        "// Source: tools/generator/kernel_traits.yaml",
        "",
    ]
    lines.extend(emit_selector(architectures, "fwd"))
    lines.extend(emit_selector(architectures, "fwd_split"))
    lines.extend(emit_switch_macro(architectures, "fwd"))
    lines.extend(emit_switch_macro(architectures, "fwd_split"))
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Generate kernel trait candidates and host fwd dispatch registry.")
    parser.add_argument("-a", "--arch", choices=("xcore1000", "xcore1500"), help="generate candidates for one architecture")
    args = parser.parse_args()

    architectures = load_database()
    validate(architectures)
    OUT_DIR.mkdir(exist_ok=True)
    if args.arch:
        output = OUT_DIR / f"kernel_traits_candidates_{args.arch}.yaml"
        output.write_text(yaml.safe_dump(kernel_candidates(architectures, args.arch), sort_keys=False))
    else:
        for arch in architectures:
            output = OUT_DIR / f"kernel_traits_candidates_{arch}.yaml"
            output.write_text(yaml.safe_dump(kernel_candidates(architectures, arch), sort_keys=False))
    write_registry(architectures)


if __name__ == "__main__":
    main()
