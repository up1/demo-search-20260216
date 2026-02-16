FROM pgvector/pgvector:pg16-trixie
WORKDIR /work
RUN apt-get update; apt-get upgrade -y
RUN apt-get install -y clang llvm-dev curl postgresql-server-dev-16 git build-essential
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN git clone https://github.com/veer66/chamkho-pg.git && \
    cd chamkho-pg && \
    RUSTFLAGS="-A dangerous_implicit_autorefs -A unnecessary_transmutes" cargo build --release && \
    ./install.sh