



clear all;
%chord length of blade assumed constant with radius
chord=0.10;

%collective angle.
collective=8.0/180*pi;

%max cyclic angle.
cyclic=0.0/180*pi;

%diameter of the rotor
dia=10.0;

%tip radius
R=dia/2.0;

%rotor speed in RPM
RPM=400.;

%thickness to chord ratio for propeller section (constant with radius)
tonc=0.12*chord;

%standard sea level atmosphere density
rho=1.225;

%RPM --> revs per sec
n=RPM/60.0;

%rps --> rads per sec
omega=n*2.0*pi;

% use 16 blade segments (starting at 20% R (hub) to 95%R)
rstep=(0.95-0.2)/16*R;

% forward velocity
V=0.0;

%tilt
tilt=0.0/180.0*pi;

% climb speed
Vc=0.0;

% max flapping velocity

vflap=0.0;
thrust=0.0;
torque=0.0;
Mx=0.0;
My=0.0;
%loop over each blade element
for i=1:16,
 rad=((.95-0.2)/16*i+0.2)*R;
 r1(i)=rad/R;
 %loop over each angular sector
 for j=1:16,
  psi=pi/8*j-pi/16;
  t1(j)=psi;
  %calculate local blade element setting angle
  theta=collective+cyclic*cos(psi);
  sigma=2.0*chord/2.0/pi/rad;
  %guess initial value of induced velocity
  Vi=10.0;
  %set logical variable to control iteration
  finished=false;
  %set iteration count and check flag
  sum=1;
  itercheck=0;
  while (~finished),
   %normal velocity components
   V0=Vi+Vc+V*sin(tilt)+vflap*rad*sin(psi);
   %disk plane velocity
   V2=omega*rad+V*cos(tilt)*sin(psi);
   %flow angle
   phi=atan2(V0,V2);
   %blade angle of attack
   alpha=theta-phi;
   % lift coefficient
   cl=6.2*alpha;
   %drag coefficient
   cd=0.008-0.003*cl+0.01*cl*cl;
   %local velocity at blade
   Vlocal=sqrt(V0*V0+V2*V2);
   %thrust grading
   DtDr=0.5*rho*Vlocal*Vlocal*2.0*chord*(cl*cos(phi)-cd*sin(phi))/16.0;
   %torque grading
   DqDr=0.5*rho*Vlocal*Vlocal*2.0*chord*rad*(cd*cos(phi)+cl*sin(phi))/16.0;
   %momentum check on induced velocity
   tem1=DtDr/(pi/4.0*rad*rho*V0);
   %stabilise iteration
   Vinew=0.9*Vi+0.1*tem1;
   if Vinew<0,
    Vinew = 0;
   end;
   %check for convergence
   if (abs(Vinew-Vi)<1.0e-5),
    finished=true;
   end;
   Vi=Vinew;
   %increment iteration count
   sum=sum+1;
   %check to see if iteration stuck
   if (sum>500),
    finished=true;
    itercheck=1;
   end;
  end;
  val(i,j)=alpha;
  thrust=thrust+DtDr*rstep;
  torque=torque+DqDr*rstep;
  Mx=Mx+rad*sin(psi)*DtDr*rstep;
  My=My+rad*cos(psi)*DtDr*rstep;
 end;
end;
for i=1:16,
 for j=1:16,
  x(i,j)=r1(i)*cos(t1(j));
  y(i,j)=r1(i)*sin(t1(j));
 end;
end;
contour(x,y,val,50);
axis equal;
thrust
torque
Mx
My

