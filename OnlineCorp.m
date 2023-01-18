function [NTear_Acc,Tear_Acct,yPost_precs,Tear_Accs,ht,oldcov,MS] = OnlineCorp(Cur_targetdata,Cur_test,Src_data,ht,Options,oldcov,MS)
    
    Num_class = Options.Num_class;
    Max_update = Options.Corp_update;
    alpha = Options.SVM_alpha;
    Max_step = Options.SVM_step;
    reg = Options.SVM_reg;
    
    Xt = Cur_targetdata(:,2:end);
    Yt = Cur_targetdata(:,1);
    Xtest = Cur_test(:,2:end);
    Ytest = Cur_test(:,1);
    [hs,Src_data,oldcov] = CORAL_update(Src_data,Xt,Yt,Options,[]);
    
    Tar_m = length(ht);
    Tar_m = Tar_m+1;
    ht{Tar_m} = SVM_train(Xt, Yt, [], Num_class, alpha, Max_step, reg);
    Tar_all_pred = predict_base(Xt,ht);
    [Cwt,Mwt] = Init_w(Tar_all_pred, Yt, [],Num_class);

    Src_m = length(Src_data);
    Src_all_pred = predict_base(Xt,hs);
    [Cws,Mws] = Init_w(Src_all_pred, Yt, [],Num_class);
    MS = [MS;Mws];

    Wtt = [];
    for i = 1:Tar_m
        Wtt = [Wtt;Cwt{i}];
    end
    Wtt = Wtt./repmat(sum(Wtt,1),Tar_m,1);
    YWt = sum(Mwt);
    Tar_all_pred = predict_base(Xt,ht);
    Post_prec = zeros(length(Yt),Num_class);
    for i = 1:Tar_m
        kk = repmat(Wtt(i,:),length(Yt),1).*Tar_all_pred{i}.soft;
        Post_prec = Post_prec+kk;
    end
    [F_max,Tar_pred]=max(Post_prec,[],2);
    Awt = (sum(Tar_pred==Yt)/length(Yt));
    

%     YWt = 1.2*sum(Mwt)
    %Update the weight of instance in source
    Selind = [];
    for i = 1:length(Mws)
        index = find(Mwt>Mws(i));
        Selind{i} = index;
    end

    Secind = find(Awt>Mws);
    [Hs,update_num,Nwt] = Corp_update(hs,ht,Cwt,Mws,Mwt,Src_data,Options,Secind,Xt,Yt);
 
%     Wtt = [];
%     for i = 1:Tar_m
%         Wtt = [Wtt;Nwt{i}];
%     end
% %     YWt = sum(mean(Wtt,2));
%     Wtt = Wtt./repmat(sum(Wtt,1),Tar_m,1);
    
    
    Tar_all_pred = predict_base(Xtest,ht);
    NPost_prec = zeros(length(Ytest),Num_class);
    for i = 1:Tar_m
        NPost_prec = NPost_prec+repmat(Wtt(i,:),length(Ytest),1).*Tar_all_pred{i}.soft;
    end
    NPost_prec = YWt*NPost_prec;   
    
    Src_all_pred = predict_base(Xtest,hs);
    yPost_precs = zeros(length(Ytest),Num_class);
    for i = 1:Src_m
        yPost_precs = yPost_precs+Mws(i)*Src_all_pred{i}.soft;
    end
    
    Post_precs = zeros(length(Ytest),Num_class);
    for i = 1:Src_m
        prec = zeros(length(Ytest),Num_class);
        for j = 1:update_num
            chs{1} = Hs{j,i};
            Src_all_pred = predict_base(Xtest,chs);
            prec = prec+Src_all_pred{1}.soft;
        end
        Post_precs = Post_precs+Mws(i)*prec/update_num;
    end
    
    [F_max,NTar_pred]=max(NPost_prec+yPost_precs,[],2);
    Tear_Acct = sum(NTar_pred==Ytest)/length(Ytest);
    
    [F_max,NTar_pred]=max(NPost_prec+Post_precs,[],2);
    NTear_Acc = sum(NTar_pred==Ytest)/length(Ytest);
    
    [F_max,NTar_pred]=max(yPost_precs,[],2);
    yPost_precs = sum(NTar_pred==Ytest)/length(Ytest);
    
    [F_max,Tar_pred]=max(NPost_prec,[],2);
    Tear_Accs = sum(Tar_pred==Ytest)/length(Ytest);
    
    if length(Mwt)>10
        [Minwtt,~] = min(Wtt(1:Tar_m,:),[],2);
        [~,remove_idx] = min(Minwtt);
        ht(remove_idx) = [];
    end
end

function Nw = Rinit_w(all_pred,label,W,Ws,Mwt,Isw,Num_class)
    for i = 1:length(Mwt)
        pre = all_pred{i}.label;
        if ~isempty(Isw)
            iswi = Isw;
        else
            iswi = ones(length(pre),1);
        end
        Auc = sum(iswi'*(pre==label))/length(label);
        Nw{i} = (Ws*Auc+W{i})/(1+Ws);
    end
end

function [CW,Auc] = Init_w(all_pred, label,Isw,Num_class)
    m = length(all_pred);
    Auc = [];
    CW = [];
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
        Auc(i) = (sum(iswi'*(pre==label))/length(label));
        z = (1-err2);
        CW{i}= z;
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

function Isw = Iw_update(Src_data,Tar_m,YWt,Ws,ht,Num_class,Wss)
    Isw = Wss;
%     Selind = 1:length(Mwt);
    for i = 1:length(Ws)
        weight = Wss{i};
        srcdata = Src_data{i};
        X = srcdata(:,2:end);
        Y = srcdata(:,1);

        Tar_all_pred = predict_base(X,ht);
        Post_prec = zeros(length(Y),Num_class);
%         sindex = Selind{sec_index};
%         for j = 1:length(Mwt)
%             Post_prec = Post_prec+Mwt(j)*Tar_all_pred{j}.soft;
%         end
%         
        Wtt = [];
        for j = 1:Tar_m
            Wtt = [Wtt;YWt{j}];
        end
        Wtt = Wtt./repmat(sum(Wtt,1),Tar_m,1);
        for j = 1:Tar_m
            Post_prec = Post_prec+Wtt(j,:).*Tar_all_pred{j}.soft;
        end
       
        [F_max,pre_y]=max(Post_prec,[],2);
        margin = (pre_y~=Y);
        Isw{i} = weight.* (Ws(i).^margin);
    end
end

function [Hs,update_num,Wt] = Corp_update(hs,ht,Wt,Ws,Mwt,Src_data,Options,Secind,Xt,Yt)
    
    Num_class = Options.Num_class;
    Max_update = Options.Corp_update;
    alpha = Options.SVM_alpha;
    Max_step = Options.SVM_step;
    reg = Options.SVM_reg;
    
    Tar_m = length(ht);
    Hs = [];
    Wss = [];
    update_num = 1;
    for i = 1:length(hs)
        for j = 1:Max_update
            Hs{j,i} = hs{i};
        end
        Wss{i} = ones(size(Src_data{i},1),1);
    end
    if ~isempty(Secind)
        update_num = Max_update;
        for Updat_k = 2:Max_update
            
            Wss = Iw_update(Src_data,Tar_m,Wt,Ws,ht,Num_class,Wss);
            [hs,Src_data,~] = CORAL_update(Src_data,Xt,Yt,Options,Wss);
            for i = 1:length(hs)
                Hs{Updat_k,i} = hs{i};
            end
%             for i = length(Src_data)
%                 srcdata = Src_data{i};
%                 X = srcdata(:,2:end);
%                 Y = srcdata(:,1);
%                 Pred_y = predict_base(X,ht);
%                 Wt = Rinit_w(Pred_y, Y, Wt, Ws(i), Mwt,Wss{i},Num_class);  %????
%             end
        end
    end
end

