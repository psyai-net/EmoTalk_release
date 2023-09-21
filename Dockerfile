FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04
MAINTAINER "Jungwoo Choi"

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

ADD requirements.txt /tmp/requirements.txt
RUN \
    # Fix CUDA apt error
    rm -f /etc/apt/sources.list.d/cuda.list  && \
    rm -f /etc/apt/sources.list.d/nvidia-ml.list  && \
    apt-get update && apt-get install -y gnupg2 software-properties-common && \
    apt-key del 7fa2af80  && \
    apt-get update && apt-get install -y --no-install-recommends wget  && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb  && \
    dpkg -i cuda-keyring_1.0-1_all.deb  && \
    # Install Start 
    apt update  && \
    add-apt-repository -y ppa:savoury1/ffmpeg4 && \
    apt -y install python3.8 python3.8-distutils libgl1-mesa-glx libglib2.0-0 git wget zsh vim openssh-server curl ffmpeg && \
    # Python Library 
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113  && \
    pip install -r /tmp/requirements.txt  && \
    # zsh option
    chsh -s /bin/zsh  && \
    sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"  && \
    # zsh-autosuggestions, zsh-syntax-highlighting을 플러그인에 추가하는 코드
    git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions  && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting  && \
    # Perl이란? : https://happygrammer.github.io/guide/perl/
    # 펄을 활용하면 vi ~/.zshrc를 해서 직접 수정해야하는 부분이 자동화가 가능하다!!
    perl -pi -w -e 's/ZSH_THEME=.*/ZSH_THEME="af-magic"/g;' ~/.zshrc  && \
    perl -pi -w -e 's/plugins=.*/plugins=(git ssh-agent zsh-autosuggestions zsh-syntax-highlighting)/g;' ~/.zshrc  && \
    # ssh에서 id:password를 설정합니다. 디폴트로 id = root, password = root으로 했습니다.
    # 보안을 위해 바꾸는 걸 추천합니다.
    # PermitRootLogin : 디폴트값을 yes로 해줘야 ssh 연결에서 문제가 안생깁니다.
    echo 'root:root' |chpasswd  && \
    sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config  && \
    sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config  && \
    mkdir /root/.ssh  && \
    mkdir /var/run/sshd   && \
    # 경우에 따라 시간대가 안맞는 에러가 발생해서, 이 코드는 웬만하면 넣는게 좋습니다.
    apt-get install -y language-pack-en && update-locale  && \
    # Clean up
    apt-get clean  && \
    apt-get autoclean  && \
    apt-get autoremove -y  && \
    rm -rf /var/lib/cache/*  && \
    rm -rf /var/lib/log/*

WORKDIR /workspace
CMD ["echo", "nvidia/cudagl:11.3.1-devel-ubuntu20.04 is ready!", 'zsh']
