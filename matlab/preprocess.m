function x = preprocess(x, thresh_in, rate_in)
    if nargin > 1
        thresh = thresh_in;
    else
        thresh = 0.2;
        rate = 1;
    end

    if nargin > 2
        rate = rate_in;
    else
        rate = 1;
    end

    disp('orig size')
    [x_orig, y_orig, z_orig] = size(x);

    [x_down, y_down, z_down] = ...
        meshgrid(0:1:x_orig-1, 0:1:y_orig-1, 0:1:z_orig-1);

    [x_up, y_up, z_up] = ...
        meshgrid(0:1/rate:x_orig-1, 0:1/rate:y_orig-1, 0:1/rate:z_orig-1);

    x = interp3(x_down, y_down, z_down, x, x_up, y_up, z_up);
    
    min_x = min(x, [], 'all');
    max_x = max(x, [], 'all');
       
    % Normalize between 0 and 1
    x = (x - min_x)/(max_x - min_x);
    
    % Linear threshold
    x(x < thresh) = 0.;

    x = x*1024;

end