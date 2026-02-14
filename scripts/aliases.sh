#!/bin/bash
# Bash alias'ları için - ~/.bashrc veya ~/.zshrc'ye eklenebilir

echo "🚀 D2L Alias'ları yükleniyor..."

# D2L proje dizini alias'ları
alias d2l='cd /home/kael/lab/d2l'
alias d2l-notebook='cd /home/kael/lab/d2l && make notebook'
alias d2l-test='cd /home/kael/lab/d2l && make test'
alias d2l-lint='cd /home/kael/lab/d2l && make lint'
alias d2l-clean='cd /home/kael/lab/d2l && make clean'

# Hızlı workflow komutları
alias d2l-quick='cd /home/kael/lab/d2l && ./scripts/quick-start.sh'
alias d2l-workflow='cd /home/kael/lab/d2l && ./scripts/workflow-helper.sh'

# CUDA komutları
alias d2l-build='cd /home/kael/lab/d2l && make cuda-build'
alias d2l-gpu='cd /home/kael/lab/d2l && make gpu-info'

# Notebook hızlı erişim
alias d2l-lab='jupyter lab --notebook-dir=/home/kael/lab/d2l/notebooks/d2l/'

# Yardım
alias d2l-help='cd /home/kael/lab/d2l && make help'

echo "✅ Alias'lar yüklendi! Yeni terminal açın veya 'source ~/.bashrc' çalıştırın."
