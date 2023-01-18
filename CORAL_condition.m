function [hs,Src_data,cov_target] = CORAL_condition(Src_data,Cur_Xt,Yt,Options,W)
    
    alpha = Options.SVM_alpha;
    Max_step = Options.SVM_step;
    reg = Options.SVM_reg;
    Num_class = Options.Num_class;
    hs = [];
    
    for i = 1:length(Src_data)
        Xs = Src_data{i}(:,2:end);
        Ys = Src_data{i}(:,1);
        Trans_X = Xs;
        L = size(Cur_Xt,2);
        for cc = 1:Num_class
            cindt = find(Yt==cc);
            cinds = find(Ys==cc);
            CXs = Xs(cinds,:);
            CXt = Cur_Xt(cindt,:);
            
            if ~isempty(W)
                CW = W{i}(cinds);
                cov_source = cov(repmat(CW,1,L).*CXs) + eye(size(repmat(CW,1,L).*CXs, 2));
            else
                cov_source = cov(CXs) + eye(size(CXs, 2));
                
            end
            cov_target = cov(CXt) + eye(size(CXt, 2));
            A_coral = cov_source^(-1/2)*cov_target^(1/2);
            CTrans_X = CXs*A_coral;
            Trans_X(cinds,:) = CTrans_X;
        end
        Src_data{i} = [Ys,Trans_X];
        
        hs{i} = SVM_train(Trans_X,Ys,[],Num_class,alpha,Max_step,reg);
    end
end