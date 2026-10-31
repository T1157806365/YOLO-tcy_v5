cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5
# 先确认当前状态 
git status 
git branch --show-current 
git remote -v 
# 基于当前代码创建新的备份分支 
git switch -c backup-20261031-teacher100
# 提交当前所有未被 .gitignore 排除的修改 
git add . 
git commit -m "Backup YOLO-tcy_v5 20261031--teacher100" 
# 上传新分支到 GitHub 
git push -u origin backup-20261031--teacher100


# 重新推送
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5
git -c http.version=HTTP/1.1 push -u origin backup-20261031-teacher


# 如果服务器已经配置 GitHub SSH 密钥，也可以尝试改走 SSH：
timeout 15 ssh -T git@github.com
# 如果 SSH 认证成功，直接推送：
git push -u git@github.com:T1157806365/YOLO-tcy_v5.git \
  backup-20261031-teacher