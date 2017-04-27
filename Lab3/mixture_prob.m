function prob = mixture_prob(image, K, L, mask, seed, Mean)
% This function uses a mask to identify pixels from an image that are used to estimate a mixture of K Gaussian
% components. The mask has the same size as the image. A pixel that is to be included has a mask value
% 1, otherwise 0. A call to the function can be written as
% prob = mixture prob(image, K, L, mask);
% where L is the number of iterations that Expectation-Maximization is supposed to run. The output of
% the function is an image of probabilities (prob) that corresponds to p(ci) in Eq. (3) above. The whole
% function can me summarized as follows:

% Note that to initialize µk, you may use K-means that you have
% already implemented and set ?k to an equal value for all components, with wk set to the fraction of
% pixels assigned to each K-means cluster. Here are some additional recommendations.

%Take into account that
%This operation builds a vector of KxMatrix empty matrixes
%cov = cell(K,Matrix);

%The pixels are reordered in a column. And each column represents a colour.
%It also converts the number to floating numbers.
%Ivec = single(reshape(I, width*height, 3));

%element by element binary operations
%diff = bsxfun(@minus, Ivec, mean);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Store all pixels for which mask = 1 in a Nx3 matrix%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%We need to transform the image in a vector, as this way the computations are faster We need the width and heigth
width = size(image,2);
height = size(image,1);
%Reshaping and changing the format
Ivec = double(reshape(image, width*height, 3)); %Three for the colour info
%To apply the mask, the format has to be the same format
mask_vec = reshape(mask,width*height,1); %one instead of three as we are not saving the colour info here
%We change Ivec to store only the pixels ci that makes mask(ci) = 1. We use
%condicional binary indexing. 
Ivec_mask = Ivec(mask_vec == 1,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Randomly initialize the K components using masked pixels%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%we need to compute the mean using K means
%Remember: function [ segmentation, centers ] = kmeans_segm(image, K, L, seed, Mean);
% where segmentation are the indexes of each cluster, and the centers, the
% value

[index, means] = kmeans_segm(Ivec_mask, K, L, seed, Mean, 0);
%Vector of K empty matrixes
sigmaK = cell(K,1);
%we asigned the value of sigma. As it sais in the lab notes 'set ?k to an
%equal value for all components'
sigmaK(:) = {diag([1 1 1])};

%we need the number of pixels that passed the mask filter
N = size(Ivec_mask,1);
w = zeros(K,1);
%We asigned the w. We use the sumatory of the number of pixels that
%are part of the current cluster and we divide by the total number of pixel
%(this way the weight will be the percentage of pixels in each cluster)
for k = 1:K
   w(k) = (1/N)*sum(index == k);
end

%%%%%%%%%%%%%%%%%%
% Iterate L times%
%%%%%%%%%%%%%%%%%%
P_ik = zeros(size(Ivec_mask,1),K);
for it = 1: L
 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Expectation: Compute probabilities P_ik using masked pixels%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %the probability has to be calculated for each cluster
    for k = 1:K 
        %to compute the operation ci - mui we use the operator proposed in
        %the lab notes. bsxfun, the does the element by element
        %substraction of the matrixes Ivec_mask and means(k,:)
        difference = bsxfun(@minus, Ivec_mask , means(k,:));
        %the distribution g is
        %We do it this way as sigmaK{k}\difference' ois the same as doing
        %inv(sigmaK{k}*difference') and we get a 3x1 matrix. Then, as
        %sum([3x1].*[3x1]) is the same as the scalar product of both
        %vectors 3x1 and 3x1, we do the escalar product this way. This
        %what the expressions of the distribution says.
        % (ci - mk)T[3x1]T*sigma^-1[3x3]*(ci - muk)[3x1] = [1x3]*[3x1] =
        % 1x1
        g_k = (1/(sqrt((2*pi)^3*det(sigmaK{k}))))*exp(-0.5*sum(difference'.*(sigmaK{k}\difference'),1));
        %Explanation of the exp: we change the order of the multiplication
        %compare to the formula of the lab notes as we are multiplying
        %matrixes directly and not numbers to make the code work faster.
        %Also, we use the sumatory, as we want the sumatorry of the three
        %values RGB of the color of each pixel. 
        %Then P_ik
        P_ik(:,k) = w(k)*g_k;
    end
    
    %we need the sumatory of P_ik
    P_ik_sum = sum(P_ik,2);
    
    %This sumatory has to be in a 1xK vector, then
    P_ik_sum_vec = repmat(P_ik_sum,1,K);
    
    %We compute the final P_ik
    P_ik = P_ik./P_ik_sum_vec;
    P_ik(isnan(P_ik)) = 0;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Maximization: Update w, means and covariances using masked pixels%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Update w
    w = (1/N)*sum(P_ik,1);
    
    for k = 1:K  
        %For each k we hace to multply the colour of each pixel by the
        %probability P_ik. Then we do 
        mult = [ P_ik(:,k) P_ik(:,k) P_ik(:,k)];
        %We compute the new values for the mean
        means(k,:)= (sum(mult.*Ivec_mask))/sum(P_ik(:,k));
    end
    
    %We repeat the calculations to obtain the maximized sigma
     %the probability has to be calculated for each cluster
    for k = 1:K 
        %to compute the operation ci - mui we use the operator proposed in
        %the lab notes. bsxfun, the does the element by element
        %substraction of the matrixes Ivec_mask and means(k,:)
        difference = bsxfun(@minus, Ivec_mask , means(k,:));
        %it is similar to the computations of the updated means
        %For each k we hace to multply the colour of each pixel by the
        %probability P_ik. Then we do 
        mult = [P_ik(:,k) P_ik(:,k) P_ik(:,k)];
%         %We compute the new values for sigma. First we do the
%         %multiplication term by term (is not the same that the
%         %multiplication of matrixes) and then we multiply the resultant
%         %matrix by differences matrix. This way it would be the same as
%         %doing for each pixel the multiplication and then add the resultan
%         %matrix for each pixel.
        sigmaK{k} = ((mult.*difference)'*difference)/sum(P_ik(:,k));
    end 
end
% End of L iterations


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute probabilities p(c_i) in Eq.(3) for all pixels I.%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
prob=zeros(size(Ivec,1),K);
 %we use the same algorithm as before but computed for every pixel not
 %onlly the masked ones
    for k = 1:K 
        %to compute the operation ci - mui we use the operator proposed in
        %the lab notes. bsxfun, the does the element by element
        %substraction of the matrixes Ivec_mask and means(k,:)
        difference = bsxfun(@minus, Ivec , means(k,:));
        %the distribution g is
        %We do it this way as sigmaK{k}\difference' ois the same as doing
        %inv(sigmaK{k}*difference') and we get a 3x1 matrix. Then, as
        %sum([3x1].*[3x1]) is the same as the scalar product of both
        %vectors 3x1 and 3x1, we do the escalar product this way. This
        %what the expressions of the distribution says.
        % (ci - mk)T[3x1]T*sigma^-1[3x3]*(ci - muk)[3x1] = [1x3]*[3x1] =
        % 1x1
        g_k = (1/(sqrt((2*pi)^3*det(sigmaK{k}))))*exp(-0.5*sum(difference'.*(sigmaK{k}\difference'),1));
        %Explanation of the exp: we change the order of the multiplication
        %compare to the formula of the lab notes as we are multiplying
        %matrixes directly and not numbers to make the code work faster.
        %Also, we use the sumatory, as we want the sumatorry of the three
        %values RGB of the color of each pixel. 
        %Then P_ik
        prob(:,k) = w(k)*g_k;
    end
    %As we did previously
    prob = sum(prob,2);
    %It has to have the same format as the original image so
    prob = reshape(prob,height,width);
end



% % function prob = mixture_prob(img, K, L, mask,a ,b)
% % % function prob = mixture prob(image, K, L, mask)
% % %L is the number of iterations that Expectation-Maximization is supposed to run.
% % % The output of the function is an image of probabilities (prob) that
% % corresponds to p(ci)
% 
% % Let I be a set of pixels and V be a set of K Gaussian components in 3D (R,G,B).
% height=size(img,1);
% width=size(img,2);
% Ivec = double(reshape(img, width*height, 3));
% maskvec = reshape(mask, width*height, 1);
% 
% % Store all pixels for which mask=1 in a Nx3 matrix
% Ivec_mask = Ivec(maskvec==1,:);
% 
% Randomly initialize the K components using masked pixels
% [init_seg,means]=kmeans_segm(Ivec_mask, K, L, 30, 2,0); % FOR TIGER1 works with seed=30.
% 
% sigma=diag([1,1,1]);
% sigma_vec=cell(K,1);
% sigma_vec(:)={sigma};
% 
% N=size(Ivec_mask,1);
% w=zeros(K,1);
% for i=1:K
%     w(i)=sum(init_seg==i)/N;
% end
% 
% P_ik=zeros(size(Ivec_mask,1),K);

% Iterate L times
% for i =1:L
% % Expectation: Compute probabilities P_ik using masked pixels
%     for k=1:K
%         D=bsxfun(@minus, Ivec_mask, means(k,:));
%         g_t=1/sqrt((2*pi)^3*det(sigmaK{k}));
%         P_ik(:,k)=w(k)*g_t*exp(-0.5*sum(D'.*(sigmaK{k}\D'),1));
% % P_ik(:,k)=w(k)*g_t*exp(-0.5*sum(D'.*(inv(sigmaK{k})*D'),1));
%     end
%     P_ik=P_ik./repmat((sum(P_ik,2)+1e-200),1,K);
% %     P_ik(isnan(P_ik))=0;
%     
% Maximization: Update w, means and covariances using masked pixels
%     w=sum(P_ik,1)/N;
%     for k=1:K
%         mean(k,:)=sum(repmat(P_ik(:,k),1,3).*Ivec_mask)/(sum(P_ik(:,k))+1e-200);
%     end
%     for k=1:K
%         D=bsxfun(@minus, Ivec_mask, means(k,:));
%         Dp=(repmat(P_ik(:,k),1,3));
%         sigmaK{k}=(mult.*difference)'*difference/sum(P_ik(:,k)+1e-200);
%         if(sigmaK{k})==0 sigmaK{k}=sigma;end
%         if(sigmaK{k})<1 sigmaK{k}=sigma;end
%     end
%     
% end
% Compute probabilities p(c_i) in Eq.(3) for all pixels I.
% prob=zeros(size(Ivec,1),K);
% for k=1:K
%     D=bsxfun(@minus, Ivec, means(k,:));
%     g_t=1/sqrt((2*pi)^3*det(sigmaK{k}));
%     prob(:,k)=w(k)*g_t*exp(-1/2*sum(D'.*(sigmaK{k}\D'),1));
% end
% prob=sum(prob,2);
% prob = reshape(prob, height,width);