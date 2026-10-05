#!/usr/bin/env bash
# Stage 0 -- project-local HIP toolchain for the llama.cpp build (ADR 0016).
#
# The host has the ROCm *runtime* (/opt/rocm-<ver>) but not hipcc, the HIP headers or
# rocm-llvm, and there is no passwordless sudo. The missing packages are downloaded from
# AMD's apt repository for the SAME ROCm version, unpacked with dpkg -x, and overlaid on a
# symlink copy of the host tree. Nothing outside vendor/rocm is modified.
set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:-$(ls -d /opt/rocm-[0-9]* 2>/dev/null | sed 's|.*/rocm-||' | sort -V | tail -1)}"
[ -n "$ROCM_VERSION" ] || { echo "FATAL: no /opt/rocm-<version> on this host." >&2; exit 1; }
HOST_ROCM="/opt/rocm-$ROCM_VERSION"
[ -d "$HOST_ROCM" ] || { echo "FATAL: $HOST_ROCM not found." >&2; exit 1; }

VENDOR="$(mkdir -p "${ROCM_VENDOR_DIR:-vendor/rocm}" && cd "${ROCM_VENDOR_DIR:-vendor/rocm}" && pwd)"
DIST="${ROCM_APT_DIST:-jammy}"
REPO="https://repo.radeon.com/rocm/apt/$ROCM_VERSION"
PACKAGES="hip-dev hipcc rocm-llvm rocm-device-libs hipblas hipblas-dev rocsolver hsa-rocr-dev rocm-cmake rocm-core"

mkdir -p "$VENDOR/debs" "$VENDOR/extract"
echo ">> host ROCm $ROCM_VERSION; fetching missing HIP toolchain from $REPO ($DIST)"

if [ ! -s "$VENDOR/Packages" ]; then
  curl -fsSL "$REPO/dists/$DIST/main/binary-amd64/Packages.gz" | gunzip > "$VENDOR/Packages.tmp" \
    || { echo "FATAL: cannot fetch $REPO/dists/$DIST/main/binary-amd64/Packages.gz" >&2; exit 1; }
  mv "$VENDOR/Packages.tmp" "$VENDOR/Packages"
fi

for pkg in $PACKAGES; do
  file="$(awk -v p="$pkg" '$1=="Package:"&&$2==p{f=1} f&&$1=="Filename:"{print $2; exit}' "$VENDOR/Packages")"
  [ -n "$file" ] || { echo "FATAL: package $pkg not in $REPO ($DIST)" >&2; exit 1; }
  deb="$VENDOR/debs/$(basename "$file")"
  if [ ! -s "$deb" ]; then
    echo ">> downloading $pkg"
    curl -fsSL -o "$deb.part" "$REPO/$file" && mv "$deb.part" "$deb"
  fi
  dpkg -x "$deb" "$VENDOR/extract"
done

echo ">> overlaying the extracted toolchain on a symlink copy of $HOST_ROCM"
rm -rf "$VENDOR/prefix"
mkdir -p "$VENDOR/prefix"
cp -rs "$HOST_ROCM"/. "$VENDOR/prefix"/
cp -rsf "$VENDOR/extract/opt/rocm-$ROCM_VERSION"/. "$VENDOR/prefix"/

export ROCM_PATH="$VENDOR/prefix" HIP_PATH="$VENDOR/prefix"
"$ROCM_PATH/bin/hipcc" --version | head -2 \
  || { echo "FATAL: hipcc from $ROCM_PATH does not run." >&2; exit 1; }

if [ -r /dev/kfd ] && [ -w /dev/kfd ]; then
  echo ">> /dev/kfd is accessible"
else
  echo "WARNING: /dev/kfd is not read/write for this user; add it to 'render' and 'video'." >&2
fi
echo ">> ROCm toolchain ready at $ROCM_PATH"
