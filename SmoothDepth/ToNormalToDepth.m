path='/data/yl/code/Sketch/Data/CUFSF1/getSameLocationCUFSF/Res/Test/';
pathD='/data/yl/code/Sketch/Data/CUFSF1/getSameLocationCUFSF/Res/TestDealSmooth/';
subpath=dir(fullfile(path,'*.mat*'));
for i=1:length(subpath)
    fileName=[path,subpath(i).name];
    depth=load(fileName);
    depth=reshape(depth.A,[256,256]);
    [nx,ny,nz]=surfnorm(depth);
    n(:,:,1)=nx;
    n(:,:,2)=ny;
    n(:,:,3)=nz;
    %%% you can change the slant parameter here, currently it's 4, usually try values between 4 and 30, it controls the depth discontinuity.
    [ Z ] = Integration_FC( n, ones(256,256), 30, 'F', 0, 0 );
    %%% display the surface
    %{
    figure;
    subplot(2,2,1);
   
    showsurf(depth);
    camlight('right');
    view(-40,40);
    title('origin depth');
    subplot(2,2,2);
    
    showsurf(Z);
    camlight('right');
    view(-40,40);
    title('depth-normal-depth');
    %%% display the original depth from Zhang
    
    subplot(2,2,3);
    
    depth2=averfilter(depth,3);
    showsurf(depth2);
    camlight('right');
    view(-40,40);
    title('Mean Value filter 3');
    
    
    subplot(2,2,4);
    %}
    depthSmooth=averfilter(depth,3);
    showsurf(depthSmooth);
    camlight('right');
    view(0,90);
    savepath=[pathD, subpath(i).name];
    %view(-40,40);
    %title('Mean Value filter 5');
    save(savepath,'depthSmooth');
    a=1;

    
end