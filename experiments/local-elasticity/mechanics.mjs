/** Independent Float64 reduced elasticity implementation.
 * FreeForm Eq. 1 (affine skinning) and Eq. 16 (stable Neo-Hookean):
 * https://research.nvidia.com/labs/sil/projects/freeform/assets/main.pdf
 * No Kaolin implementation is incorporated. Volumes/materials are caller data;
 * sample-based integration does not establish physical calibration or full
 * surface contact. Floor contact is frictionless at the integration samples.
 */
export const MECHANICS_LIMITS=Object.freeze({points:512,handles:6,iterations:64,evaluations:256,milliseconds:1000});
const dot=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;};
const norm=a=>Math.sqrt(dot(a,a));
const finite=(a,length,name)=>{if(!a||a.length!==length||Array.from(a).some(x=>!Number.isFinite(x)))throw Error(`Invalid ${name}`);return Float64Array.from(a);};
const materialArray=(value,n,name)=>finite(typeof value==='number'?Array(n).fill(value):value,n,name);
const now=()=>globalThis.performance?.now?.()??Date.now();

/** Jacobi diagonalization of a small real symmetric matrix; vectors are columns. */
export function symmetricEigen(matrix,n,{tolerance=1e-14,maxSweeps=100}={}){
  const a=finite(matrix,n*n,'symmetric matrix'),v=new Float64Array(n*n);let sweeps=0;
  for(let i=0;i<n;i++){v[i*n+i]=1;for(let j=0;j<i;j++)if(Math.abs(a[i*n+j]-a[j*n+i])>1e-10*Math.max(1,Math.abs(a[i*n+j])))throw Error('Matrix is not symmetric');}
  const scale=Math.max(...Array.from(a,Math.abs));
  for(;sweeps<maxSweeps;sweeps++){
    let largest=0;
    for(let p=0;p<n;p++)for(let q=p+1;q<n;q++){
      const apq=a[p*n+q];largest=Math.max(largest,Math.abs(apq));if(Math.abs(apq)<=tolerance*scale)continue;
      const tau=(a[q*n+q]-a[p*n+p])/(2*apq),t=(tau>=0?1:-1)/(Math.abs(tau)+Math.hypot(1,tau));
      const c=1/Math.hypot(1,t),s=t*c,app=a[p*n+p],aqq=a[q*n+q];
      a[p*n+p]=app-t*apq;a[q*n+q]=aqq+t*apq;a[p*n+q]=a[q*n+p]=0;
      for(let k=0;k<n;k++)if(k!==p&&k!==q){const x=a[k*n+p],y=a[k*n+q];a[k*n+p]=a[p*n+k]=c*x-s*y;a[k*n+q]=a[q*n+k]=s*x+c*y;}
      for(let k=0;k<n;k++){const x=v[k*n+p],y=v[k*n+q];v[k*n+p]=c*x-s*y;v[k*n+q]=s*x+c*y;}
    }
    if(largest<=tolerance*scale)break;
  }
  const order=Array.from({length:n},(_,i)=>i).sort((i,j)=>a[j*n+j]-a[i*n+i]),values=new Float64Array(n),vectors=new Float64Array(n*n);
  let residual=0;
  for(let k=0;k<n;k++){values[k]=a[order[k]*n+order[k]];for(let i=0;i<n;i++)vectors[i*n+k]=v[i*n+order[k]];}
  for(let k=0;k<n;k++)for(let i=0;i<n;i++){let x=0;for(let j=0;j<n;j++)x+=matrix[i*n+j]*vectors[j*n+k];residual=Math.max(residual,Math.abs(x-values[k]*vectors[i*n+k]));}
  if(residual>Math.max(Number.MIN_VALUE,scale)*1e-10)throw Error('Mass eigensystem failed its residual check');
  return {values,vectors,sweeps,residual,relativeResidual:scale?residual/scale:0};
}

function lame(young,poisson){
  if(young<0||!Number.isFinite(young)||!Number.isFinite(poisson)||poisson<=-.99||poisson>=.499)throw Error('Material requires E>=0 and -.99<Poisson<.499');
  const mu=young/(2*(1+poisson)),lambda=young*poisson/((1+poisson)*(1-2*poisson)),lambdaBar=lambda+mu;
  return {mu,lambda,lambdaBar,gamma:lambdaBar?1+mu/lambdaBar:1};
}
function cofactor(F){return new Float64Array([
  F[4]*F[8]-F[5]*F[7],F[5]*F[6]-F[3]*F[8],F[3]*F[7]-F[4]*F[6],
  F[2]*F[7]-F[1]*F[8],F[0]*F[8]-F[2]*F[6],F[1]*F[6]-F[0]*F[7],
  F[1]*F[5]-F[2]*F[4],F[2]*F[3]-F[0]*F[5],F[0]*F[4]-F[1]*F[3]]);}

/** Row-major F; derivative is dPsi/dF, also row-major. Valid at singular F. */
export function stableNeoHookean(F,young,poisson){
  finite(F,9,'deformation gradient');const p=lame(young,poisson),C=cofactor(F),J=F[0]*C[0]+F[1]*C[1]+F[2]*C[2];
  const E0=p.lambdaBar*(1-p.gamma)**2+3*p.mu;
  const energy=.5*(p.lambdaBar*(J-p.gamma)**2+p.mu*dot(F,F)-E0);
  const gradient=Float64Array.from(F,(x,j)=>p.mu*x+p.lambdaBar*(J-p.gamma)*C[j]);
  return {energy,gradient,determinant:J,cofactor:C,...p};
}

function cholesky(A,n){
  const L=new Float64Array(n*n),scale=Math.max(...Array.from({length:n},(_,i)=>Math.abs(A[i*n+i])));
  for(let i=0;i<n;i++)for(let j=0;j<=i;j++){
    let x=A[i*n+j];for(let k=0;k<j;k++)x-=L[i*n+k]*L[j*n+k];
    if(i===j){if(!(x>1e-14*scale)||!Number.isFinite(x))return null;L[i*n+j]=Math.sqrt(x);}else L[i*n+j]=x/L[j*n+j];
  }return L;
}
function cholSolve(L,b,n){
  const x=Float64Array.from(b);
  for(let i=0;i<n;i++){for(let j=0;j<i;j++)x[i]-=L[i*n+j]*x[j];x[i]/=L[i*n+i];}
  for(let i=n-1;i>=0;i--){for(let j=i+1;j<n;j++)x[i]-=L[j*n+i]*x[j];x[i]/=L[i*n+i];}return x;
}
function multiply(A,x,n){const out=new Float64Array(n);for(let i=0;i<n;i++)for(let j=0;j<n;j++)out[i]+=A[i*n+j]*x[j];return out;}

// Exact linear equality handling. Normalize rows, remove dependent equations
// only after checking RHS consistency, and keep the QR factors for dual signs.
function constraintQR(rows,n){
  const Q=[],rhs=[],selected=[],R=[];let redundant=0;
  for(const source of rows){
    const length=norm(source.a);if(length<1e-14){if(Math.abs(source.b)>1e-9)throw Error('Inconsistent zero constraint');redundant++;continue;}
    const a=Float64Array.from(source.a,x=>x/length),b=source.b/length,v=Float64Array.from(a),r=new Float64Array(Q.length+1);let beta=b;
    for(let pass=0;pass<2;pass++)for(let j=0;j<Q.length;j++){const t=dot(v,Q[j]);r[j]+=t;beta-=t*rhs[j];for(let k=0;k<n;k++)v[k]-=t*Q[j][k];}
    const remainder=norm(v);
    if(remainder<1e-8){if(Math.abs(beta)>1e-7)throw Error('Dependent pins/contact equations are inconsistent');redundant++;continue;}
    r[Q.length]=remainder;Q.push(Float64Array.from(v,x=>x/remainder));rhs.push(beta/remainder);R.push(r);selected.push({...source,a,b,normalization:length});
  }
  const Z=[];
  for(let j=0;j<n&&Z.length<n-Q.length;j++){
    const v=new Float64Array(n);v[j]=1;
    for(let pass=0;pass<2;pass++)for(const row of [...Q,...Z]){const t=dot(v,row);for(let k=0;k<n;k++)v[k]-=t*row[k];}
    const d=norm(v);if(d>1e-8)Z.push(Float64Array.from(v,x=>x/d));
  }
  if(Z.length!==n-Q.length)throw Error('Constraint nullspace construction failed');
  return {Q,rhs,R,Z,selected,redundant};
}
function newtonDirection(H,g,qr){
  const n=g.length,m=qr.Z.length,freeGradient=Float64Array.from(qr.Z,z=>dot(z,g)),A=new Float64Array(m*m);
  const Hz=qr.Z.map(z=>multiply(H,z,n));
  for(let i=0;i<m;i++)for(let j=0;j<m;j++)A[i*m+j]=dot(qr.Z[i],Hz[j]);
  let shift=0,L=m?cholesky(A,m):new Float64Array(0);
  const scale=Math.max(1,...Array.from({length:m},(_,i)=>Math.abs(A[i*m+i])));
  for(let attempt=0;!L&&attempt<14;attempt++){
    shift=attempt===0?scale*1e-8:shift*10;const regularized=Float64Array.from(A);for(let i=0;i<m;i++)regularized[i*m+i]+=shift;L=cholesky(regularized,m);
  }
  if(!L)throw Error('Unable to obtain a positive definite Newton model');
  const free=m?cholSolve(L,Float64Array.from(freeGradient,x=>-x),m):new Float64Array(0),direction=new Float64Array(n);
  for(let j=0;j<m;j++)for(let i=0;i<n;i++)direction[i]+=qr.Z[j][i]*free[j];
  const force=multiply(H,direction,n);for(let i=0;i<n;i++)force[i]+=g[i]+shift*direction[i];
  const dual=Float64Array.from(qr.Q,q=>-dot(q,force));
  // A_selected = R Q, hence R^T lambda = -Q(g+Hp).
  for(let i=dual.length-1;i>=0;i--){for(let j=i+1;j<dual.length;j++)dual[i]-=qr.R[j][i]*dual[j];dual[i]/=qr.R[i][i];}
  return {direction,dual,shift,projectedGradient:norm(freeGradient)};
}

export function createMechanics({restPositions,volumes,weights,gradients,density=1000,young=1000,poisson=.3,massRelativeTolerance=1e-10}){
  const N=restPositions?.length/3,H=weights?.length/N,S=4*H,D=12*H;
  if(!Number.isInteger(N)||N<4||N>MECHANICS_LIMITS.points||!Number.isInteger(H)||H<1||H>MECHANICS_LIMITS.handles)throw Error('Mechanics point/handle budget exceeded');
  if(!(massRelativeTolerance>=1e-12&&massRelativeTolerance<=1e-6))throw Error('Invalid explicit mass rank tolerance');
  const X=finite(restPositions,3*N,'rest positions'),V=finite(volumes,N,'volumes'),W=finite(weights,N*H,'weights'),DW=finite(gradients,N*H*3,'weight gradients');
  const rho=materialArray(density,N,'density'),E=materialArray(young,N,'Young modulus'),nu=materialArray(poisson,N,'Poisson ratio');
  if(Array.from(V).some(v=>v<=0)||Array.from(rho).some(v=>v<=0))throw Error('Volumes and densities must be positive');
  const materials=Array.from({length:N},(_,i)=>lame(E[i],nu[i])),masses=Float64Array.from(V,(v,i)=>v*rho[i]);
  const B=new Float64Array(N*S),G=new Float64Array(N*S*3),M=new Float64Array(S*S);
  for(let i=0;i<N;i++)for(let h=0;h<H;h++)for(let a=0;a<4;a++){
    const k=4*h+a,x=a<3?X[3*i+a]:1;B[i*S+k]=W[i*H+h]*x;
    for(let j=0;j<3;j++)G[(i*S+k)*3+j]=DW[(i*H+h)*3+j]*x+(a===j?W[i*H+h]:0);
  }
  for(let i=0;i<N;i++)for(let a=0;a<S;a++)for(let b=0;b<S;b++)M[a*S+b]+=masses[i]*B[i*S+a]*B[i*S+b];
  const eigen=symmetricEigen(M,S),largest=eigen.values[0],rank=Array.from(eigen.values).filter(v=>v>largest*massRelativeTolerance).length;
  if(!(largest>0)||rank===0||eigen.values.at(-1)<-largest*1e-10)throw Error('Invalid mass Gram matrix');
  const T=new Float64Array(S*rank),BT=new Float64Array(N*rank),GT=new Float64Array(N*rank*3),MT=new Float64Array(rank*rank);
  for(let a=0;a<S;a++)for(let k=0;k<rank;k++)T[a*rank+k]=eigen.vectors[a*S+k]/Math.sqrt(eigen.values[k]);
  for(let i=0;i<N;i++)for(let k=0;k<rank;k++)for(let a=0;a<S;a++){
    BT[i*rank+k]+=B[i*S+a]*T[a*rank+k];for(let j=0;j<3;j++)GT[(i*rank+k)*3+j]+=G[(i*S+a)*3+j]*T[a*rank+k];
  }
  for(let i=0;i<N;i++)for(let a=0;a<rank;a++)for(let b=0;b<rank;b++)MT[a*rank+b]+=masses[i]*BT[i*rank+a]*BT[i*rank+b];
  let whiteningError=0;for(let a=0;a<rank;a++)for(let b=0;b<rank;b++)whiteningError=Math.max(whiteningError,Math.abs(MT[a*rank+b]-(a===b?1:0)));
  if(whiteningError>2e-6)throw Error('Mass whitening validation failed');
  const diagnostics={points:N,handles:H,fullDofs:D,reducedDofs:3*rank,scalarMassRank:rank,droppedScalarModes:S-rank,
    massRelativeTolerance,massEigenvalues:Array.from(eigen.values),massEigenResidual:eigen.relativeResidual,
    retainedMassCondition:largest/eigen.values[rank-1],whiteningError,totalMass:Array.from(masses).reduce((a,b)=>a+b,0),
    rankScope:'Discarded modes are null/negligible only in this sampled mass metric; their behavior away from these samples is not certified.'};
  function internal(q){q=finite(q,D,'affine coordinates');const c=new Float64Array(D);for(let h=0;h<H;h++)for(let r=0;r<3;r++)for(let a=0;a<4;a++)c[r*S+4*h+a]=q[h*12+r*4+a];return c;}
  function external(c){const q=new Float64Array(D);for(let h=0;h<H;h++)for(let r=0;r<3;r++)for(let a=0;a<4;a++)q[h*12+r*4+a]=c[r*S+4*h+a];return q;}
  function toReduced(q){const c=internal(q),z=new Float64Array(3*rank);for(let r=0;r<3;r++)for(let k=0;k<rank;k++)for(let a=0;a<S;a++)z[r*rank+k]+=Math.sqrt(eigen.values[k])*eigen.vectors[a*S+k]*c[r*S+a];return z;}
  function fromReduced(z){z=finite(z,3*rank,'reduced coordinates');const c=new Float64Array(D);for(let r=0;r<3;r++)for(let a=0;a<S;a++)for(let k=0;k<rank;k++)c[r*S+a]+=T[a*rank+k]*z[r*rank+k];return external(c);}
  function pointData(spec,reduced){
    let x,row;
    if(Number.isInteger(spec.index)){
      if(spec.index<0||spec.index>=N)throw Error('Point index outside integration samples');
      x=X.slice(3*spec.index,3*spec.index+3);row=B.slice(spec.index*S,(spec.index+1)*S);
    }else{
      x=finite(spec.restPosition,3,'constraint rest position');const w=finite(spec.weights,H,'constraint weights');row=new Float64Array(S);
      for(let h=0;h<H;h++)for(let a=0;a<4;a++)row[4*h+a]=w[h]*(a<3?x[a]:1);
    }
    if(reduced){const next=new Float64Array(rank);for(let a=0;a<S;a++)for(let k=0;k<rank;k++)next[k]+=row[a]*T[a*rank+k];row=next;}
    return {x,row,target:finite(spec.target,3,'constraint target')};
  }
  function evaluate(c,{basis=B,derivative=G,mass=M,size=S,predicted=null,dt=1,gravity=[0,0,0],drag=null,hessian=false}={}){
    const n=3*size,g=new Float64Array(n),A=hessian?new Float64Array(n*n):null,positions=new Float64Array(3*N),Fs=new Float64Array(9*N);
    gravity=finite(gravity,3,'gravity');if(!(dt>0&&Number.isFinite(dt)))throw Error('dt must be positive');
    let elasticEnergy=0,inertialEnergy=0,gravityEnergy=0,dragEnergy=0,minDeterminant=Infinity,energyComponentMagnitude=0;
    for(let i=0;i<N;i++){
      const F=new Float64Array([1,0,0,0,1,0,0,0,1]);
      for(let r=0;r<3;r++){
        let x=X[3*i+r];for(let k=0;k<size;k++){x+=basis[i*size+k]*c[r*size+k];for(let j=0;j<3;j++)F[r*3+j]+=derivative[(i*size+k)*3+j]*c[r*size+k];}
        positions[3*i+r]=x;gravityEnergy-=masses[i]*gravity[r]*x;energyComponentMagnitude+=Math.abs(masses[i]*gravity[r]*x);
      }
      Fs.set(F,9*i);const C=cofactor(F),J=F[0]*C[0]+F[1]*C[1]+F[2]*C[2],{mu,lambdaBar,gamma}=materials[i],a=lambdaBar*(J-gamma),P=new Float64Array(9);
      minDeterminant=Math.min(minDeterminant,J);
      elasticEnergy+=V[i]*.5*(lambdaBar*((J-gamma)**2-(1-gamma)**2)+mu*(dot(F,F)-3));
      // Magnitudes before the rest-energy subtractions are needed near an
      // equilibrium, where the final energy can hide cancellation. This is an
      // explicit floating-point allowance scale, not a rigorous error bound.
      energyComponentMagnitude+=V[i]*.5*(lambdaBar*((J-gamma)**2+(1-gamma)**2)+mu*(dot(F,F)+3));
      for(let j=0;j<9;j++)P[j]=mu*F[j]+a*C[j];
      const cg=hessian?new Float64Array(3*size):null;
      for(let r=0;r<3;r++)for(let k=0;k<size;k++){
        let value=0,cof=0;for(let j=0;j<3;j++){const d=derivative[(i*size+k)*3+j];value+=P[r*3+j]*d;cof+=C[r*3+j]*d;}
        g[r*size+k]+=V[i]*value-masses[i]*gravity[r]*basis[i*size+k];if(cg)cg[r*size+k]=cof;
      }
      if(A)for(let k=0;k<size;k++)for(let l=0;l<size;l++){
        const o=(i*size+k)*3,p=(i*size+l)*3,x=derivative[o],y=derivative[o+1],z=derivative[o+2],u=derivative[p],v=derivative[p+1],w=derivative[p+2];
        const gg=x*u+y*v+z*w,cross=[y*w-z*v,z*u-x*w,x*v-y*u];
        for(let r=0;r<3;r++)for(let s=0;s<3;s++){
          let value=lambdaBar*cg[r*size+k]*cg[s*size+l];
          if(r===s)value+=mu*gg;
          else {const row=3-r-s,sign=(r===0&&s===1)||(r===1&&s===2)||(r===2&&s===0)?1:-1;value+=a*sign*(cross[0]*F[row*3]+cross[1]*F[row*3+1]+cross[2]*F[row*3+2]);}
          A[(r*size+k)*n+s*size+l]+=V[i]*value;
        }
      }
    }
    if(predicted){
      finite(predicted,n,'predicted coordinates');const factor=1/(dt*dt);
      for(let r=0;r<3;r++)for(let k=0;k<size;k++)for(let l=0;l<size;l++){
        const a=r*size+k,b=r*size+l,m=mass[k*size+l]*factor,d=c[b]-predicted[b];
        const contribution=.5*(c[a]-predicted[a])*m*d;
        inertialEnergy+=contribution;energyComponentMagnitude+=Math.abs(contribution);g[a]+=m*d;if(A)A[a*n+b]+=m;
      }
    }
    if(drag){
      if(!Number.isFinite(drag.stiffness)||drag.stiffness<0||drag.stiffness>1e12)throw Error('Drag stiffness must be finite in [0,1e12]');
      const p=pointData(drag,size===rank&&basis===BT);
      for(let r=0;r<3;r++){
        let difference=p.x[r]-p.target[r];for(let k=0;k<size;k++)difference+=p.row[k]*c[r*size+k];
        dragEnergy+=.5*drag.stiffness*difference*difference;
        energyComponentMagnitude+=.5*drag.stiffness*difference*difference;
        for(let k=0;k<size;k++){g[r*size+k]+=drag.stiffness*difference*p.row[k];if(A)for(let l=0;l<size;l++)A[(r*size+k)*n+r*size+l]+=drag.stiffness*p.row[k]*p.row[l];}
      }
    }
    const energy=elasticEnergy+inertialEnergy+gravityEnergy+dragEnergy;
    if(!Number.isFinite(energy)||Array.from(g).some(x=>!Number.isFinite(x))||A&&Array.from(A).some(x=>!Number.isFinite(x)))throw Error('Nonfinite mechanics evaluation');
    return {energy,gradient:g,hessian:A,positions,deformationGradients:Fs,elasticEnergy,inertialEnergy,gravityEnergy,dragEnergy,minDeterminant,energyComponentMagnitude};
  }
  function energyGradient(q,options={}){
    const {dt=1,gravity=[0,0,0],drag=null,hessian=false}=options;
    const next={dt,gravity,drag,hessian,predicted:options.predicted?internal(options.predicted):null},e=evaluate(internal(q),next);
    e.gradient=external(e.gradient);
    if(e.hessian){const old=e.hessian,out=new Float64Array(D*D);for(let h=0;h<H;h++)for(let c=0;c<3;c++)for(let a=0;a<4;a++)for(let j=0;j<H;j++)for(let d=0;d<3;d++)for(let b=0;b<4;b++)out[(12*h+4*c+a)*D+12*j+4*d+b]=old[(c*S+4*h+a)*D+d*S+4*j+b];e.hessian=out;}
    return e;
  }
  function evaluatePositions(q,{positions=X,weights=W}={}){
    const count=positions.length/3;if(!Number.isInteger(count)||count>250000)throw Error('Visible point budget exceeded');
    finite(positions,3*count,'query positions');finite(weights,count*H,'query weights');q=finite(q,D,'affine coordinates');
    const out=Float64Array.from(positions);
    for(let i=0;i<count;i++)for(let h=0;h<H;h++)for(let r=0;r<3;r++){
      let u=q[12*h+4*r+3];for(let a=0;a<3;a++)u+=q[12*h+4*r+a]*positions[3*i+a];out[3*i+r]+=weights[i*H+h]*u;
    }return out;
  }
  function step(state,{dt=1/60,gravity=[0,-9.81,0],pins=[],drag=null,floor=null,maxIterations=24,maxEvaluations=160,maxMilliseconds=100,gradientTolerance=1e-6}={}){
    if(!(dt>=1e-5&&dt<=.1)||!Number.isInteger(maxIterations)||maxIterations<1||maxIterations>64||!Number.isInteger(maxEvaluations)||maxEvaluations<2||maxEvaluations>256||!(maxMilliseconds>0&&maxMilliseconds<=1000)||!(gradientTolerance>0&&gradientTolerance<=.01))throw Error('Invalid bounded integration controls');
    if(!Array.isArray(pins)||pins.length>64||floor!==null&&!Number.isFinite(floor))throw Error('Invalid hard constraints');
    const started=now(),q0=finite(state.q,D,'state q'),v0=finite(state.velocity,D,'state velocity'),z0=toReduced(q0),velocity=toReduced(v0),n=3*rank;
    let z=Float64Array.from(z0);const predicted=Float64Array.from(z0,(v,i)=>v+dt*velocity[i]);
    const pinRows=[];
    for(const pin of pins){const p=pointData(pin,true);for(let r=0;r<3;r++){const a=new Float64Array(n);a.set(p.row,r*rank);pinRows.push({a,b:p.target[r]-p.x[r],kind:'pin',index:pin.index??null});}}
    const pinQR=constraintQR(pinRows,n);
    for(let i=0;i<pinQR.Q.length;i++){const correction=pinQR.rhs[i]-dot(pinQR.Q[i],z);for(let k=0;k<n;k++)z[k]+=pinQR.Q[i][k]*correction;}
    const floorRows=floor===null?[]:Array.from({length:N},(_,i)=>{const a=new Float64Array(n);a.set(BT.subarray(i*rank,(i+1)*rank),rank);return {a,b:floor-X[3*i+1],kind:'floor',index:i};});
    const constraintError=x=>({pin:Math.max(0,...pinRows.map(r=>Math.abs(dot(r.a,x)-r.b))),floor:Math.max(0,...floorRows.map(r=>r.b-dot(r.a,x)))});
    if(constraintError(z).floor>1e-8)throw Error('Initial state/pins penetrate the floor; a feasible start is required');
    const active=new Set(floorRows.filter(r=>dot(r.a,z)-r.b<1e-8).map(r=>r.index));
    const history=[];let evaluations=0,status='iteration-limit',accepted=0,maximumShift=0,projectedGradient=Infinity,contactDualViolation=Infinity,roundoffAcceptedSteps=0,largestRoundoffAllowance=0;
    const options={basis:BT,derivative:GT,mass:MT,size:rank,predicted,dt,gravity,drag};
    let e=evaluate(z,{...options,hessian:true});evaluations++;
    const initialEnergy=e.energy;
    for(let iteration=0;iteration<maxIterations;iteration++){
      if(now()-started>maxMilliseconds){status='time-limit';break;}
      const qr=constraintQR([...pinRows,...Array.from(active,i=>floorRows[i])],n),direction=newtonDirection(e.hessian,e.gradient,qr);
      maximumShift=Math.max(maximumShift,direction.shift);projectedGradient=direction.projectedGradient;
      let release=-1,worst=gradientTolerance;
      for(let i=0;i<qr.selected.length;i++)if(qr.selected[i].kind==='floor'&&direction.dual[i]>worst){worst=direction.dual[i];release=qr.selected[i].index;}
      contactDualViolation=release<0?0:worst;
      if(projectedGradient<=gradientTolerance){
        if(release>=0){active.delete(release);history.push({iteration,event:'release-contact',index:release});continue;}
        status='converged';break;
      }
      const descent=dot(e.gradient,direction.direction);
      if(!(descent<0)){status='no-descent';break;}
      let alpha=1,blocker=-1;
      for(const row of floorRows)if(!active.has(row.index)){
        const change=dot(row.a,direction.direction);if(change<0){const limit=Math.max(0,(dot(row.a,z)-row.b)/-change);if(limit<alpha){alpha=limit;blocker=row.index;}}
      }
      if(alpha<1e-12&&blocker>=0){active.add(blocker);history.push({iteration,event:'add-contact',index:blocker});continue;}
      let trial,trialEvaluation,stepAccepted=false,acceptanceReason=null,roundoffAllowance=0,trialProjectedGradient=null;
      for(let backtrack=0;backtrack<24;backtrack++){
        if(evaluations>=maxEvaluations||now()-started>maxMilliseconds)break;
        trial=Float64Array.from(z,(v,i)=>v+alpha*direction.direction[i]);
        try{trialEvaluation=evaluate(trial,options);evaluations++;}catch{evaluations++;trialEvaluation=null;}
        const errors=trialEvaluation?constraintError(trial):null;
        if(trialEvaluation&&errors.floor<=1e-8&&errors.pin<=1e-8){
          if(trialEvaluation.energy<=e.energy+1e-4*alpha*descent){stepAccepted=true;acceptanceReason='armijo';break;}
          // When the predicted decrease is itself below energy roundoff, a
          // strict energy comparison can reject an accurate Newton step.
          // Require the SAME active-set nullspace gradient to halve instead;
          // feasibility and the original convergence threshold still apply.
          roundoffAllowance=64*Number.EPSILON*(e.energyComponentMagnitude+trialEvaluation.energyComponentMagnitude);
          if(Number.isFinite(roundoffAllowance)&&-alpha*descent<=roundoffAllowance&&trialEvaluation.energy<=e.energy+roundoffAllowance){
            trialProjectedGradient=Math.hypot(...qr.Z.map(row=>dot(row,trialEvaluation.gradient)));
            if(trialProjectedGradient<=.5*projectedGradient){stepAccepted=true;acceptanceReason='energy-roundoff-gradient-reduction';break;}
          }
        }
        alpha*=.5;
      }
      if(!stepAccepted){status=evaluations>=maxEvaluations?'evaluation-limit':now()-started>maxMilliseconds?'time-limit':'line-search-failed';break;}
      const historyEntry={iteration,energy:e.energy,nextEnergy:trialEvaluation.energy,alpha,projectedGradient,shift:direction.shift,acceptanceReason};
      if(acceptanceReason==='energy-roundoff-gradient-reduction'){
        roundoffAcceptedSteps++;largestRoundoffAllowance=Math.max(largestRoundoffAllowance,roundoffAllowance);
        Object.assign(historyEntry,{roundoffAllowance,trialProjectedGradient,predictedDecrease:-alpha*descent,
          energyComponentMagnitude:e.energyComponentMagnitude,trialEnergyComponentMagnitude:trialEvaluation.energyComponentMagnitude});
      }
      history.push(historyEntry);z=trial;accepted++;
      if(blocker>=0&&dot(floorRows[blocker].a,z)-floorRows[blocker].b<1e-8)active.add(blocker);
      if(evaluations>=maxEvaluations){e=trialEvaluation;status='evaluation-limit';break;}
      e=evaluate(z,{...options,hessian:true});evaluations++;
    }
    const q=fromReduced(z),nextVelocity=Float64Array.from(q,(v,i)=>(v-q0[i])/dt),errors=constraintError(z);
    const projection=evaluatePositions(fromReduced(z0)),original=evaluatePositions(q0);let initialProjectionError=0;
    for(let i=0;i<projection.length;i++)initialProjectionError=Math.max(initialProjectionError,Math.abs(projection[i]-original[i]));
    return {state:{q,velocity:nextVelocity},diagnostics:{status,converged:status==='converged',acceptedIterations:accepted,evaluations,
      initialEnergy,finalEnergy:e.energy,elasticEnergy:e.elasticEnergy,minDeterminant:e.minDeterminant,maximumNewtonShift:maximumShift,
      projectedGradient,contactDualViolation,pinError:errors.pin,floorPenetration:errors.floor,activeFloorSamples:active.size,
      roundoffAcceptedSteps,largestRoundoffAllowance,
      roundoffAcceptanceRule:'Only after Armijo rejects: predicted decrease and energy increase within 64 eps times both evaluations\' pre-cancellation energy-component magnitudes; same active-nullspace gradient at most half; unchanged feasibility and convergence tolerances.',
      stationarityScope:'Projected gradient and contact dual signs describe the last Newton linearization; at a finite solver limit they need not describe the returned accepted point.',
      initialMassProjectionPointError:initialProjectionError,completedWallMs:now()-started,history,
      solver:'Mass-whitened active-set regularized Newton with Armijo line search; finite solver limits are reported',
      constraints:'Exact linear pins and frictionless floor at integration samples; drag is an explicit finite-stiffness point spring'}};
  }
  return {pointCount:N,handleCount:H,dofs:D,reducedDofs:3*rank,diagnostics,
    createState:()=>({q:new Float64Array(D),velocity:new Float64Array(D)}),toReduced,fromReduced,energyGradient,evaluatePositions,step,
    massMatrix:Float64Array.from(M),masses:Float64Array.from(masses)};
}
