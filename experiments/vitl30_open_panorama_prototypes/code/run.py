"""Resumable GPU-only runner; status files describe actual running stages."""
import argparse
import fcntl
import traceback
from .data import OUT, prepare, status
from .train import train

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--capacity',type=int,default=96)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            prepare(workers=args.workers)
            result = train(device=args.device,epochs=args.epochs,capacity=args.capacity)
            from .export import export
            gallery = export(device=args.device)
            status(OUT,'complete',templates=result['templates'],metrics=result['dev'],gallery=gallery,
                detail=f'首轮完成：{result["templates"]}个完整全景候选模板；开发集组级重建已评分，语义与稳定性待核查。')
        except BaseException as exc:
            status(OUT,'failed',error=str(exc),detail='运行中断，保留检查点；'+str(exc))
            traceback.print_exc()
            raise

if __name__=='__main__':
    main()
