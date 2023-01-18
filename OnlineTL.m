function [Tear_Acc,CORAL_A,ht,Tear_Accs,Tear_Acct] = OnlineTL(Cur_targetdata,Cur_test,Src_data,Src_origX, hs,ht,Options,CORAL_A)
    
    Num_class = Options.Num_class;
    Max_update = Options.Corp_update;
    alpha = Options.SVM_alpha;
    Max_step = Options.SVM_step;
    reg = Options.SVM_reg;
    
    Xt = Cur_targetdata(:,2:end);
    Yt = Cur_targetdata(:,1);
    Tar_num = length(Yt);
    Src_data = CORAL_update(Src_origX,Xt,Src_data,[]);
    % update CORAL
    
    Tar_m = length(ht);
    Tar_m = Tar_m+1;
    ht{Tar_m} = My_SVM(Xt, Yt, [], Num_class, alpha, Max_step, reg); %???? concept ???
    Wt = ones(Tar_m,1);
    Tar_all_pred = predict_base(Xt,ht);
    [Wt,Terr,auct] = calculate_w(Tar_all_pred, Yt, Wt,Num_class,[]);  %????
    Aterr = sum(auct);
    
    Src_m = length(Src_data);
    Ws = ones(Src_m,1);
    Src_all_pred = predict_base(Xt,hs);
    [Ws,Serr,aucs] = calculate_w(Src_all_pred, Yt, Ws,Num_class,[]);
    
    Ytest = Cur_test(:,1);
    Post_prec = zeros(length(Ytest),Num_class);
    Tar_all_pred = predict_base(Cur_test(:,2:end),ht);
    
    Wtt = [];
    for i = 1:Tar_m
        Wtt = [Wtt;Wt{i}];
    end
    
    Wts = [];
    for i = 1:Src_m
        Wts = [Wts;Ws{i}];
    end

    Wtt = Wtt./repmat(sum(Wtt,1),Tar_m,1);
    for i = 1:Tar_m
        Post_prec = Post_prec+Wtt(i,:).*Tar_all_pred{i}.soft;
    end
    Post_prec = Post_prec*Aterr;
    
    Post_precs = zeros(length(Ytest),Num_class);
    Tar_all_pred = predict_base(Cur_test(:,2:end),hs);
    for i = 1:Src_m
        Post_precs = Post_precs+aucs(i).*Tar_all_pred{i}.soft;
    end
    
    [F_max,Tar_pred]=max(Post_prec,[],2);
    Tear_Acct = sum(Tar_pred==Ytest)/length(Ytest);
    
    [F_max,Tar_pred]=max(Post_precs,[],2);
    Tear_Accs = sum(Tar_pred==Ytest)/length(Ytest);
    
    [F_max,Tar_pred]=max(Post_prec+Post_precs,[],2);
    Tear_Acc = sum(Tar_pred==Ytest)/length(Ytest);
    
    if length(Wt)>10
        [Minwtt,~] = min(Wtt(1:Tar_m,:),[],2);
        [~,remove_idx] = min(Minwtt);
        ht(remove_idx) = [];
    end
end

function [W,Err,auc] = calculate_w(all_pred, label, W, Num_class,Isw)
    m = length(all_pred);
    Err = [];
    W = [];
    %     Num_class=10;
    for i=1:m
        err2 = zeros(1,Num_class);
        pre = all_pred{i}.label;
        if ~isempty(Isw)
            iswi = Isw;
        else
            iswi = ones(length(pre),1);
        end
        
        for j = 1:length(label)
            err2(label(j)) = err2(label(j))+iswi(j)'*(pre(j)~=label(j));
        end
        for j = 1: Num_class
            err2(j) = err2(j)/length(find(label==j));
        end
        %         err2 = sum(err2)/Num_class;
        auc(i) = sum(iswi'*(pre==label))/length(label);
        Err{i} = err2;
        z = (1-err2);
        W{i}= z;
    end
end

function pred = predict_base(data,model)  %??????concept????????
    m = length(model);
    pred = [];
    for i=1:m
        clf = model{i};
        [label,Posterior] = SVM_pre(data,clf);
        pred{i}.soft = Posterior;
        pred{i}.label = label;
    end
end



