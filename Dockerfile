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
    # add zsh-autosuggestions, zsh-syntax-highlighting plugin
    git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions  && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting  && \
    # Modify .zshrc whth Perl
    perl -pi -w -e 's/ZSH_THEME=.*/ZSH_THEME="af-magic"/g;' ~/.zshrc  && \
    perl -pi -w -e 's/plugins=.*/plugins=(git ssh-agent zsh-autosuggestions zsh-syntax-highlighting)/g;' ~/.zshrc  && \
    # Set ssh id and password, default is id = root, password = root.
    # I recommand changing this for more security
    # PermitRootLogin : yes - for ssh connection
    echo 'root:root' |chpasswd  && \
    sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config  && \
    sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config  && \
    mkdir /root/.ssh  && \
    mkdir /var/run/sshd   && \
    # install language pack for timeline issue.
    apt-get install -y language-pack-en && update-locale  && \
    # Clean up
    apt-get clean  && \
    apt-get autoclean  && \
    apt-get autoremove -y  && \
    rm -rf /var/lib/cache/*  && \
    rm -rf /var/lib/log/*

WORKDIR /workspace
CMD ["echo", "nvidia/cudagl:11.3.1-devel-ubuntu20.04 is ready!", 'zsh']
