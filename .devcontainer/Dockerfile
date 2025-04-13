FROM texlive/texlive:latest

RUN apt update && apt upgrade -y && apt install -y build-essential git
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo install tex-fmt
RUN tlmgr init-usertree
RUN tlmgr install texliveonfly