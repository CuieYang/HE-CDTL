function [Tear_Acc,ht] = OnlineICL(Cur_targetdata,Cur_test,ht,Options)

Num_class = Options.Num_class;
Max_update = Options.Corp_update;
alpha = Options.SVM_alpha;
Max_step = Options.SVM_step;
reg = Options.SVM_reg;

Xt = Cur_targetdata(:,2:end);
Yt = Cur_targetdata(:,1);
Tar_num = length(Yt);

Tar_m = length(ht);

Tar_m = Tar_m+1;
ht{Tar_m} = My_SVM(Xt, Yt, [], Num_class, alpha, Max_step, reg); %???? concept ???
Wt = ones(Tar_m,1);
Tar_all_pred = predict_base(Xt,ht);
[Wt,Terr] = calculate_w(Tar_all_pred, Yt, Wt,[]);  %????

Ytest = Cur_test(:,1);
Post_prec = zeros(length(Ytest),Num_class);
Tar_all_pred = predict_base(Cur_test(:,2:end),ht);
for i = 1:Tar_m
    Post_prec = Post_prec+Wt(i)*Tar_all_pred{i}.soft;
end
% Post_prec = Post_prec/sum(Wt);


[F_max,Tar_pred]=max(Post_prec,[],2);
Tear_Acc = sum(Tar_pred==Ytest)/length(Ytest);


if length(Wt)>10
    [~,remove_idx] = min(Wt);
    ht(remove_idx) = [];
end

end


function [W,Err] = calculate_w(all_pred, label, W, Isw)
m = length(all_pred);
Err = [];
for i=1:m
    pre = all_pred{i}.label;
    if ~isempty(Isw)
       iswi = Isw;
    else
       iswi = ones(length(pre),1); 
    end
    err = sum(iswi'*(pre~=label))/length(label);
    Err(i) = err;
    z = (1-err);
    W(i)= 0.5*(W(i)+z);
end

% for i = 1:m
%    W(i) = 1/(Err(i)+Err(end)+0.00005);
% end
% W(end) = 1/(Err(end)+0.00005);
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



