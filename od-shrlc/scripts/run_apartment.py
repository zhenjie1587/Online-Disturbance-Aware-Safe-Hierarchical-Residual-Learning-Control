"""Run the apartment interface with the reported 0.995 forgetting factor."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(ROOT/'reproduction/revision_v3'))
    import run_apartment_transfer as adapter
    # Preserved adapter source has a later 0.998 default; the reported completed trial used 0.995.
    adapter.RLS_FORGETTING=.995
    adapter.run(a.output.resolve())
if __name__=='__main__':main()
