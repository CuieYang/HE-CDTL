function [Tear_Acc,ht] = OnlineICLsep(Cur_targetdata,Cur_test,ht,Options)
    
    Num_class = Options.Num_class;
    Max_update = Options.Corp_update;
    alpha = Options.SVM_alpha;
    Max_step = Options.SVM_step;
    reg = Options.SVM_reg;
    
    Xt = Cur_targetdata(:,2:end);
    Yt = Cur_targetdata(:,1);
    Tar_m = length(ht);
    
    Tar_m = Tar_m+1;
    ht{Tar_m} = My_SVM(Xt, Yt, [], Num_class, alpha, Max_step, reg);
    Wt = ones(Tar_m,1);
    Tar_all_pred = predict_base(Xt,ht);
    [Wt,Terr,err] = calculate_w(Tar_all_pred, Yt, Wt,Num_class,[]);
    
    Ytest = Cur_test(:,1);
    Post_prec = zeros(length(Ytest),Num_class);
    Tar_all_pred = predict_base(Cur_test(:,2:end),ht);
    Wtt = [];
    for i = 1:Tar_m
        Wtt = [Wtt;Wt{i}];
    end
    Wtt = Wtt./repmat(sum(Wtt,1),Tar_m,1);
    for i = 1:Tar_m
        Post_prec = Post_prec+Wtt(i,:).*Tar_all_pred{i}.soft;
    end
    
    [F_max,Tar_pred]=max(Post_prec,[],2);
    Tear_Acc = sum(Tar_pred==Ytest)/length(Ytest);
    
    
    Post_prec1 = zeros(length(Ytest),Num_class);
    for i = 1:Tar_m
        Post_prec1 = Post_prec1+(1-err(i))*Tar_all_pred{i}.soft;
    end
    
    [F_max1,Tar_pred1]=max(Post_prec1,[],2);
    Tear_Acc1 = sum(Tar_pred1==Ytest)/length(Ytest);
    
    if length(Wt)>10
        [Minwtt,~] = min(Wtt,[],2);
%         Minwtt = sum(Wtt,2);
        [~,remove_idx] = min(Minwtt);
%         [~,remove_idx] = min(sum(Wt{i}));
        ht(remove_idx) = [];
    end
end


function [W,Err,err] = calculate_w(all_pred, label, W, Num_class,Isw)
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
        err(i) = sum(iswi'*(pre~=label))/length(label);
        Err{i} = err2;
        z = (1-err2);
        W{i}= 0.5*(ones(1,Num_class)+z);
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



