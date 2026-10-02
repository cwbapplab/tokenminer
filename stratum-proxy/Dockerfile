# Build the stratum proxy as a small, non-root runtime image.
#
# The full Rust image is used for the build stage because the dependency tree includes crates
# with C components (ring) that need a working toolchain.

FROM rust:1-bookworm AS build

WORKDIR /src

# Copy manifests first so dependency compilation is cached across source changes.
COPY Cargo.toml Cargo.lock ./
COPY protocol/Cargo.toml protocol/
COPY api-client/Cargo.toml api-client/
COPY proxy/Cargo.toml proxy/

RUN mkdir -p protocol/src api-client/src proxy/src \
    && echo '' > protocol/src/lib.rs \
    && echo '' > api-client/src/lib.rs \
    && echo 'fn main() {}' > proxy/src/main.rs \
    && cargo build --release --locked -p sp-proxy

COPY . .

RUN cargo build --release --locked -p sp-proxy

FROM debian:bookworm-slim AS runtime

RUN useradd --create-home --shell /usr/sbin/nologin proxy

COPY --from=build /src/target/release/sp-proxy /usr/local/bin/sp-proxy

USER proxy

ENV STRATUM_LISTEN_ADDR=0.0.0.0:3333
ENV RUST_LOG=info

EXPOSE 3333

ENTRYPOINT ["/usr/local/bin/sp-proxy"]
