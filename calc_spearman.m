formatSpec = '%f';
truth_file = fopen('score.txt','r');
truth = fscanf(truth_file,formatSpec);
rgb_file = fopen('test1_rgb.txt','r');
opticalflow_file = fopen('test1_opticalflow.txt','r');
rgb = fscanf(rgb_file,formatSpec);
opticalflow = fscanf(opticalflow_file,formatSpec);

r1 = corr(rgb,truth,'type','spearman');
r2 = corr(opticalflow,truth,'type','spearman');
fusion = 0.6*rgb+0.4*opticalflow;
r3 = corr(fusion,truth,'type','spearman');

