function [hs,Src_data,cov_target] = CORAL_update(Src_data,Cur_Xt,Yt,Options,W)
    
    alpha = Options.SVM_alpha;
    Max_step = Options.SVM_step;
    reg = Options.SVM_reg;
    Num_class = Options.Num_class;
    hs = [];
    
    for i = 1:length(Src_data)
        L = size(Cur_Xt,2);
        Xs = Src_data{i}(:,2:end);
        Ys = Src_data{i}(:,1);
        if ~isempty(W)
            CW = W{i};
            cov_source = cov(repmat(CW,1,L).*Xs) + eye(size(repmat(CW,1,L).*Xs, 2));
        else
            cov_source = cov(Xs) + eye(size(Xs, 2));

        end
        cov_target = cov(Cur_Xt) + eye(size(Cur_Xt, 2));
        A_coral = cov_source^(-1/2)*cov_target^(1/2);
        Trans_X = Xs*A_coral;
        Src_data{i} = [Ys,Trans_X];
%         if ~isempty(W)
%             hs{i} = SVM_train(Trans_X,Ys,W{i},Num_class,alpha,Max_step,reg);
%         else
            hs{i} = SVM_train(Trans_X,Ys,[],Num_class,alpha,Max_step,reg);
%         end
    end
end