function [src_clf,Tra_Xs,Xs,Ys,A_coral] = CORAL_clf(Src_Data,Target_Data,Options)

Source_Data = Src_Data;
ID = randperm(size(Source_Data,1));

alpha = Options.SVM_alpha;
Max_step = Options.SVM_step;
reg = Options.SVM_reg;
Num_class = Options.Num_class;

% load(sprintf('data/%s', target_domain));
% Target_Data = data(:,:);

Xs = double(Source_Data(:,2:end));
Cur_Xt = double(Target_Data(:,2:end));
Ys = Source_Data(:,1);
Cur_Yt = Target_Data(:,1);

cov_source = cov(Xs) + eye(size(Xs, 2));
cov_target = cov(Cur_Xt) + eye(size(Cur_Xt, 2));
A_coral = cov_source^(-1/2)*cov_target^(1/2);

Tra_Xs = Xs * A_coral;
src_clf = My_SVM(Tra_Xs(ID,:),Ys(ID),[],Num_class,alpha,Max_step,reg);
end