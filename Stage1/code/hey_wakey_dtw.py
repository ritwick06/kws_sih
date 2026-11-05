import sys
import numpy as np
import librosa

SR=16000
N_MFCC=13

def features(path):
    y,_=librosa.load(path,sr=SR,mono=True)
    m=librosa.feature.mfcc(y=y,sr=SR,n_mfcc=N_MFCC)
    d=librosa.feature.delta(m)
    d2=librosa.feature.delta(m,order=2)
    x=np.vstack([m,d,d2]).T
    return (x-x.mean(0))/(x.std(0)+1e-8)

def dtw(a,b):
    n,m=len(a),len(b)
    D=np.full((n+1,m+1),np.inf); D[0,0]=0
    for i in range(1,n+1):
        for j in range(1,m+1):
            c=np.linalg.norm(a[i-1]-b[j-1])
            D[i,j]=c+min(D[i-1,j],D[i,j-1],D[i-1,j-1])
    return D[n,m]/(n+m)

def score(query,refs):
    q=features(query)
    return min(dtw(q,features(r)) for r in refs)

if __name__=="__main__":
    if len(sys.argv)<3:
        print("Usage: python hey_wakey_dtw.py query.wav ref1.wav [ref2.wav ...]")
        raise SystemExit
    s=score(sys.argv[1],sys.argv[2:])
    threshold=3.97636
    print(f"DTW score: {s:.4f}")
    print("TRIGGER" if s<=threshold else "REJECT")
