path= '.\data\';  % put the filepath here
subdirpath = fullfile( path, '*.mat' );  
dat = dir( subdirpath );
for j = 1  : size(dat,1)
    datpath = fullfile( path, dat( j ).name); 
    name = strsplit(dat( j ).name,'.');
    name = name{1};
    fft_all (datpath, name);
end