nohup python -u finetune.py --cfg='v8n-mup' > endovis-v8n-mup-aug.log 2>&1 &
nohup python -u finetune.py --cfg='v8n-all-1p1' > endovis-v8n-all-1p1-42-aug.log 2>&1 &
nohup python -u finetune.py --cfg='v8n-all-1p2' > endovis-v8n-all-1p2-42-aug.log 2>&1 &
nohup python -u finetune.py --cfg='v8n-all-3p2' > endovis-v8n-all-3p2-42-aug.log 2>&1 &

nohup python -u finetune.py --cfg='v8n-all-post' > endovis-v8n-all-1p2-42-post-2.log 2>&1 &

nohup python -u finetune.py --cfg='v8n-all-post' > endovis-v8n-all-post.log 2>&1 &

nohup python -u finetune.py --cfg='v8n-mup-post' > endovis-v8n-mup-post2.log 2>&1 &