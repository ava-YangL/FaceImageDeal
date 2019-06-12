function showsurf( z, x, y )
%UNTITLED1 Summary of this function goes here
%  Detailed explanation goes here
if (nargin==1)
    z1=z;
    z(size(z,1):-1:1,1:size(z,2))=z1;
    surf(z,'FaceColor',[1 1 1],'AmbientStrength',0,'DiffuseStrength',1,'SpecularStrength',0.4,'LineStyle','none','FaceLighting','flat');
else
    surf(x,y,z,'FaceColor',[1 1 1],'AmbientStrength',0,'DiffuseStrength',1,'SpecularStrength',0,'LineStyle','none','FaceLighting','phong');
end    
axis equal
%light('Position',[0 0 1],'Style','infinite');
%light('Position',[0 1 0],'Style','infinite');
