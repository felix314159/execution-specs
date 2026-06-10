#
# Each client has its
# own dedicated Dockerfile (Dockerfile.<client>) that is built into a tagged
# image.
#
# Per-client images consumed here (the builder tags each as
# steel/<client>:<branch>, NOT :latest, so several branches can coexist):
#   steel/go-ethereum:<branch>   (Dockerfile.go-ethereum)
#   steel/erigon:<branch>        (Dockerfile.erigon)
#   steel/nimbus-el:<branch>     (Dockerfile.nimbus-el)
#   steel/evmone:<branch>        (Dockerfile.evmone)
#   steel/reth:<branch>          (Dockerfile.reth)
#   steel/nethermind:<branch>    (Dockerfile.nethermind)
#   steel/besu:<branch>          (Dockerfile.besu)
#
# Which branch each is assembled from is controlled by the *_image build args
# below; each defaults to the branch pinned in clients.yaml. To assemble a
# different branch, build that branch first (edit clients.yaml, then run
# `consume direct --docker.client-branches`) then pass e.g.
# --build-arg geth_image=steel/go-ethereum:my-branch.
#
#---------------------------------------------------------------
# Final image
#---------------------------------------------------------------
FROM debian:testing-slim

# Per-client source images to COPY --from. Each defaults to the branch pinned
# in clients.yaml (tagged by the builder as steel/<client>:<branch>). Override
# any of them to assemble a different already-built branch, e.g.
#   --build-arg geth_image=steel/go-ethereum:my-feature
ARG geth_image=steel/go-ethereum:master
ARG erigon_image=steel/erigon:main
ARG nimbus_image=steel/nimbus-el:master
ARG evmone_image=steel/evmone:master
ARG reth_image=steel/reth:main
ARG nethermind_image=steel/nethermind:master
ARG besu_image=steel/besu:main

# Runtime dependencies for the bundled client binaries:
#   - libssl-dev:           required by nethtest
#   - openjdk-21-jre:       required by besu's evmtool
#   - libstdc++-13-dev:     required by evmone (GLIBC_2.38)
RUN apt-get update -q && \
    apt-get install -qy --no-install-recommends \
        ca-certificates \
        libssl-dev \
        openjdk-21-jre-headless \
        libstdc++-13-dev \
        jq nano \
  && apt-get clean

# go-ethereum
COPY --from=${geth_image} /gethvm /gethvm
ENV GETH_BIN=/gethvm

# Erigon
COPY --from=${erigon_image} /erigon_vm /erigon_vm
ENV ERIG_BIN=/erigon_vm

# Nimbus-eth1
COPY --from=${nimbus_image} /nimbvm /txparse.nimbus /
ENV NIMB_BIN=/nimbvm

# evmone
COPY --from=${evmone_image} /evmone-statetest /evmone
COPY --from=${evmone_image} /libevmone.so.* /lib/
ENV EVMO_BIN=/evmone

# reth / revme
COPY --from=${reth_image} /revme /revme
ENV RETH_BIN=/revme

# Nethermind
COPY --from=${nethermind_image} /neth /neth
RUN ln -s /neth/nethtest /nethtest
ENV NETH_BIN=/neth/nethtest

# Besu
COPY --from=${besu_image} /evmtool /evmtool
RUN ln -s /evmtool/bin/evmtool /besu-vm
ENV BESU_BIN=/evmtool/bin/evmtool

ENTRYPOINT ["/bin/bash"]
