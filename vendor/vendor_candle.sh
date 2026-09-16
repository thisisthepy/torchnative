#!/bin/sh
# Materialise the `candle-core` fork that gives this crate `DType::I8`, or check
# that the committed copy is exactly that fork.
#
#   sh vendor/vendor_candle.sh           write  vendor/candle-core/
#   sh vendor/vendor_candle.sh --check   rebuild it in a temp dir and diff
#
# The fork is two inputs and nothing else (docs/numerics/INT8.md §1.2):
#
#   candle-core 0.11.0 as crates.io published it, pinned by sha256 below. That
#   hash is the `checksum` crates.io's index records for the package -- the one
#   develop's `Cargo.lock` carried before the fork -- so it is cargo's own pin,
#   not one chosen here. `.cargo_vcs_info.json` inside it names candle commit
#   31f35b1, the same one docs/design/CANDLE_DEPS.md cloned.
#
#   vendor/int8-candle-0.11.0-cpu.patch, applied with `git apply`.
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
dest=${TORCHNATIVE_CANDLE_DIR:-$repo/vendor/candle-core}

version=0.11.0
sha256=5ecb245093b0f791b89d3420c3df9c6d49c60ab63ba54db896bf8a3baf486706
url=https://static.crates.io/crates/candle-core/candle-core-$version.crate
patch_file=$repo/vendor/int8-candle-$version-cpu.patch

mode=write
case "${1:-}" in
    '') ;;
    --check) mode=check ;;
    *) echo "usage: $0 [--check]" >&2; exit 2 ;;
esac

work=$(mktemp -d "${TMPDIR:-/tmp}/vendor-candle.XXXXXX")
trap 'rm -rf "$work"' EXIT
crate=$work/candle-core-$version.crate

# Where the .crate comes from does not matter, because nothing is trusted until
# its hash matches. In order: an explicit file (the test uses this to show a
# wrong crate is refused), cargo's own download cache, then crates.io.
if [ -n "${TORCHNATIVE_CANDLE_CRATE:-}" ]; then
    cp "$TORCHNATIVE_CANDLE_CRATE" "$crate"
    origin=$TORCHNATIVE_CANDLE_CRATE
else
    origin=
    for cached in "${CARGO_HOME:-$HOME/.cargo}"/registry/cache/*/candle-core-$version.crate; do
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
vendor_candle.sh: refusing -- candle-core-$version.crate has the wrong sha256.

  from      $origin
  expected  $sha256
  got       $got

This is not the crate the fork was made from. Nothing was written.
EOF
    exit 1
fi
echo "candle-core $version sha256 $got verified ($origin)"

tar -xzf "$crate" -C "$work"
tree=$work/candle-core-$version

# The ceiling keeps `git apply` from discovering a repository above $work (a
# TMPDIR inside a checkout) and reinterpreting the paths against its root.
if ! (cd "$tree" && GIT_CEILING_DIRECTORIES=$work git apply --whitespace=nowarn "$patch_file"); then
    echo "vendor_candle.sh: $patch_file does not apply to candle-core $version" >&2
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
vendor_candle.sh --check: $dest is NOT candle-core $version + $(basename "$patch_file").

Every edit to the fork belongs in the patch; the committed tree is regenerated
from it with \`sh vendor/vendor_candle.sh\`. The difference:
EOF
        cat "$work/drift.txt" >&2
        exit 1
    fi
    echo "$dest is candle-core $version + $(basename "$patch_file"), byte for byte"
    exit 0
fi

mkdir -p "$dest"
rsync -a --delete "$tree/" "$dest/"
echo "wrote $dest"
