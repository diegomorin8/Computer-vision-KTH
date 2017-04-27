function [ segmentation, centers ] = kmeans_segm(image, K, L, seed, Mean, resize, check_ite, graph);
%Image - in image
%K - number of clusters
%L - number of iterations
%Seed - of the random number sequence so we can repeat the sequence again. 
%Mean - Flag to determine the type of random mean generation. 2 - Mean
%generation, 1- max generation, 0 - all range generation 
%resize = 1 - resize 0 - not resize
if (nargin < 5)
    Mean = 2; 
    resize = 1;
    check_ite = 0;
    graph = 0;
elseif nargin < 6
    resize = 1;
    check_ite = 0;
    graph = 0;
elseif nargin < 7
    check_ite = 0;
    graph = 0;
elseif nargin < 8
    graph = 0;
end

threshold = 1;  %Convergence threshold 


if resize == 1
    % Let X be a set of pixels and V be a set of K cluster centers in 3D (R,G,B).
    %First, we change the format of the image is said in the lab notes
    imageD = double(image);
    %Get the dimensions
    height = size(image,1);
    width = size(image,2);
    %Reshape the image as in the lab notes
    imageD = reshape( imageD, width*height, 3);
else
    imageD = image;
end
%Band width threhold if we want a smaller band width ( between 0 and 1)
Band_width_Thres = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Randomly initialize the K cluster centers%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%We need to generate 8 color values for each cluster. 
%Remember that 
%rand([M,N]) returns an M-by-N matrix. The generated numbers are between 0
%and 1.We need then to multiply this number or the color will be always
%between 0 and 1. One option would be to generate random values of the
%mean around the mean colour of the image. Other option would be to
%generate this numbers in the whole range of possible colours. Let compare
%both options
%Option 1: random numbers around the mean
if Mean == 2
    %We need the mean value of our image, this is
    mean_Value = mean(imageD,1); %that is a 1x3 vector
    %We get the max to calculate the Band_width
    max_mean = max(mean_Value);
    min_mean = min(mean_Value);
    %Caculate the band width
    Band_width = min(Band_width_Thres*(256 - max_mean),Band_width_Thres*(0 -min_mean));
    %We need a Kx3 vector, so we just repeat this values 8 times
    mean_Value_Mat = repmat(mean_Value,K,1);
    %Use the seed
    rng(seed);
    %Generate the centers of the klusters. We need a matrix of Kx3 (Kx (R G B))
    Kmeans = mean_Value_Mat - Band_width + rand([K,3])*Band_width*2;
    %This generates Kmeans random values around the max mean value of the
    %image within a selected band width.
    
%Option 2: generate a random number between 0 and the max value of the
%image
elseif Mean == 1
    %We need the mean value of our image, this is
    max_Value = max(imageD(:))
    %Use the seed
    rng(seed);
    %Generate the centers of the klusters. We need a matrix of Kx3 (Kx (R G B))
    Kmeans = rand([K,3])*max_Value;
    
    
%Option 3: random numbers in the whole range of colours.
else
    max_Pixel_Value = 255;
    %Use the seed
    rng(seed);
    %Generate the centers of the klusters. We need a matrix of Kx3 (Kx (R G B))
    Kmeans = rand([K,3])*max_Pixel_Value;
    %This generates Kmeans random values in the whole colour range
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute all distances between pixels and cluster centers%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%We use the proposed function to get the distance from each point to each
%Kmeans. 
distances_vec = pdist2(Kmeans, imageD);
%Es una matriz de 8xwidth*height. Indice i, la observacion i de Kmeans y el
%j la observacion j de imageD y distances_vec(i,j) = distancia entre ambas
%observaciones. 

if check_ite == 1
    dist_previous = norm(distances_vec);
end


%%%%%%%%%%%%%%%%%%
% Iterate L times%
%%%%%%%%%%%%%%%%%%

for i=1:L
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Assign each pixel to the cluster center for which the dist is minimum%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %We need to assign the pixel to the cluster with minimum distances. 
    %We know that 
    % [Y,I] = min(X,[],DIM) operates along the dimension DIM.
    % We want to analyze every pixel distance so in this case we need min
    % to operate along the 1 dimension. Y wpuld be the value of the min
    % distance but we dont need it. And I would be the index with the min
    % distance, in this case, the Kluster with the mim distance, so this
    % will be out segmentation. 
    [~,segmentation] = min(distances_vec,[],1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Recompute each cluster center by taking the mean of all pixels%
    % assigned to it%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for center = 1:K
        %For every center, we calculate the new mean of the pixels assigned
        %to it.  We use logical indexing to use less steps
        Kmeans(center,:) = mean( imageD( segmentation == center,:),1);
    end
    
    % If a cluster ends with no pixels, we give the cluster a random color
    if sum(isnan(Kmeans(:)))~= 0
        %We get the id number of the clusters without any pixel associated
        id = find( isnan(Kmeans(:,1)) == 1 );
        Kmeans(id',:) = rand([size(id),3])*256;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Recompute all distances between pixels and cluster centers%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %We use the same as we used previously
    distances_vec = pdist2(Kmeans, imageD);
    
    
    %Check convergence
    if check_ite == 1
        dif = abs((dist_previous - norm(distances_vec)));
        dif_vect(i) = dif;
        if dif <= threshold
            if graph == 0
                disp(sprintf( 'Convergence at L = %i',i))
                break;
            end
        else
            dist_previous = norm(distances_vec);
            %sprintf( ' distance dif = %f \n',dif)
        end
   end
end

if check_ite == 1
    %Display of the differencesdisp(sprintf( dif ))
    disp(sprintf( 'Distance differences: '))
    for i = 1:L
        disp(sprintf( ' %i',dif_vect(i)))
    end
    if graph == 1
        plot((1:L),dif_vect)
    end
end

if resize == 1
    %The output need to be in the format of the image so
    segmentation = reshape( segmentation, height, width);
end
%Final assigning
centers = Kmeans;
end

