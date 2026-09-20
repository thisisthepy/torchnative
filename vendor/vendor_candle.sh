#!/bin/sh
# Materialise the candle forks that give this crate `DType::I8`, or check that
# the committed copies are exactly those forks.
#
#   sh vendor/vendor_candle.sh           write  vendor/candle-core/
#                                              and vendor/candle-metal-kernels/
#   sh vendor/vendor_candle.sh --check   rebuild both in a temp dir and diff
#
# TWO crates, because `DType::I8` is two halves that are useless apart. The
# `candle-core` half can build an int8 buffer on Metal; the
# `candle-metal-kernels` half is the shader symbol every kernel then looks up
# by name. docs/devices/matrix.md §7.13 measured what happens with only the
# first: 284 `operands` failures become 284 kernel-stage refusals and no cell
# moves to AGREES. Neither crate is vendored on its own.
#
# Each fork is two inputs and nothing else (docs/numerics/INT8.md §1.2):
#
#   candle-core 0.11.0 as crates.io published it, pinned by sha256 below. That
#   hash is the `checksum` crates.io's index records for the package -- the one
#   develop's `Cargo.lock` carried before the fork -- so it is cargo's own pin,
#   not one chosen here. `.cargo_vcs_info.json` inside it names candle commit
#   31f35b1, the same one docs/design/CANDLE_DEPS.md cloned.
#
#   vendor/int8-candle-0.11.0-cpu.patch, applied with `git apply`. (The name
#   says `cpu` for where the work started; the file has carried Metal counters
#   and now Metal `I8` for some time. It is quoted by docs/numerics/INT8.md
#   §1.2 and rust/torch_c/pytests/test_int8.py, so it is left alone.)
#
#   candle-metal-kernels 0.11.0, pinned by the sha256 the lock already
#   recorded, plus vendor/int8-candle-metal-kernels-0.11.0.patch.
#
# The *published* crate rather than a clone of candle's repository, on purpose:
# `cargo package` normalises `Cargo.toml`, so the published crate stands alone.
# The repository's `candle-core/Cargo.toml` inherits `version.workspace = true`
# and friends and cannot be lifted out of its 24 MB workspace -- which is why
# CANDLE_DEPS.md §9 declined to vendor it. The published crate is 1.9 MB.
#
# The result is COMMITTED, unlike vendor_torch.sh's tree. `[patch.crates-io]`
# is read before anything else runs, so a gitignored fork would need this
# script in front of every cargo invocation there is -- install_shim.sh, run.sh,
# both CI workflows' dozen cross builds, cargo-ndk, the device scripts -- and
# the first one missed fails on every machine but the one that ran it, which is
# the defect this replaces. Committed, a fresh clone builds with plain `cargo
# build`. The cost, a copy that could drift from its two inputs, is what
# `--check` answers, and rust/torch_c/pytests/test_int8.py runs it in the gate.
set -eu

repo=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)

version=0.11.0

mode=write
case "${1:-}" in
    '') ;;
    --check) mode=check ;;
    *) echo "usage: $0 [--check]" >&2; exit 2 ;;
esac

# Each crate is one line: name, sha256, patch basename. `TORCHNATIVE_CANDLE_DIR`
# and `TORCHNATIVE_CANDLE_CRATE` stay scoped to candle-core, which is what
# test_int8.py's wrong-crate refusal test drives; the second crate has its own
# pair so a test can drive either without disturbing the other.
crates="candle-core candle-metal-kernels"

sha256_for() {
    case "$1" in
    candle-core)
        echo 5ecb245093b0f791b89d3420c3df9c6d49c60ab63ba54db896bf8a3baf486706 ;;
    candle-metal-kernels)
        echo 242e83c6acf639bb273c929d73c67a882bb4dd08a140f121096e19ba2f213d3e ;;
    *) echo "sha256_for: unknown crate $1" >&2; exit 2 ;;
    esac
}

patch_for() {
    case "$1" in
    candle-core)          echo "$repo/vendor/int8-candle-$version-cpu.patch" ;;
    candle-metal-kernels) echo "$repo/vendor/int8-candle-metal-kernels-$version.patch" ;;
    *) echo "patch_for: unknown crate $1" >&2; exit 2 ;;
    esac
}

dest_for() {
    case "$1" in
    candle-core)          echo "${TORCHNATIVE_CANDLE_DIR:-$repo/vendor/candle-core}" ;;
    candle-metal-kernels) echo "${TORCHNATIVE_CANDLE_KERNELS_DIR:-$repo/vendor/candle-metal-kernels}" ;;
    esac
}

override_for() {
    case "$1" in
    candle-core)          echo "${TORCHNATIVE_CANDLE_CRATE:-}" ;;
    candle-metal-kernels) echo "${TORCHNATIVE_CANDLE_KERNELS_CRATE:-}" ;;
    esac
}

work=$(mktemp -d "${TMPDIR:-/tmp}/vendor-candle.XXXXXX")
trap 'rm -rf "$work"' EXIT

for name in $crates; do
    sha256=$(sha256_for "$name")
    patch_file=$(patch_for "$name")
    dest=$(dest_for "$name")
    override=$(override_for "$name")
    url=https://static.crates.io/crates/$name/$name-$version.crate
    crate=$work/$name-$version.crate

    # Where the .crate comes from does not matter, because nothing is trusted
    # until its hash matches. In order: an explicit file (the test uses this to
    # show a wrong crate is refused), cargo's own download cache, then crates.io.
    if [ -n "$override" ]; then
        cp "$override" "$crate"
        origin=$override
    else
        origin=
        for cached in "${CARGO_HOME:-$HOME/.cargo}"/registry/cache/*/$name-$version.crate; do
            if [ -f "$cached" ]; then
                cp "$cached" "$crate"
                origin=$cached
                break
            fi
        done
        if [ -z "$origin" ]; then
            curl -fsSL --retry 3 -o "$crate" "$url"
            origin=$url
        fi
    fi

    if command -v sha256sum >/dev/null 2>&1; then
        got=$(sha256sum "$crate" | cut -d' ' -f1)
    else
        got=$(shasum -a 256 "$crate" | cut -d' ' -f1)
    fi
    if [ "$got" != "$sha256" ]; then
        cat >&2 <<EOF
vendor_candle.sh: refusing -- $name-$version.crate has the wrong sha256.

  from      $origin
  expected  $sha256
  got       $got

This is not the crate the fork was made from. Nothing was written.
EOF
        exit 1
    fi
    echo "$name $version sha256 $got verified ($origin)"

    tar -xzf "$crate" -C "$work"
    tree=$work/$name-$version

    # The ceiling keeps `git apply` from discovering a repository above $work (a
    # TMPDIR inside a checkout) and reinterpreting the paths against its root.
    if ! (cd "$tree" && GIT_CEILING_DIRECTORIES=$work git apply --whitespace=nowarn "$patch_file"); then
        echo "vendor_candle.sh: $patch_file does not apply to $name $version" >&2
        exit 1
    fi
    echo "applied $(basename "$patch_file")"

    if [ "$mode" = check ]; then
        if [ ! -d "$dest" ]; then
            echo "vendor_candle.sh --check: $dest does not exist -- run sh vendor/vendor_candle.sh" >&2
            exit 1
        fi
        if ! diff -r "$tree" "$dest" > "$work/drift.txt"; then
            cat >&2 <<EOF
vendor_candle.sh --check: $dest is NOT $name $version + $(basename "$patch_file").

Every edit to the fork belongs in the patch; the committed tree is regenerated
from it with \`sh vendor/vendor_candle.sh\`. The difference:
EOF
            cat "$work/drift.txt" >&2
            exit 1
        fi
        echo "$dest is $name $version + $(basename "$patch_file"), byte for byte"
    else
        mkdir -p "$dest"
        rsync -a --delete "$tree/" "$dest/"
        echo "wrote $dest"
    fi
done
