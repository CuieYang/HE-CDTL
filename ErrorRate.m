function  ER = ErrorRate(LabelR, LabelH,Weight)

% ER ��������Ĵ�����
% weight ����Ȩ��
% LabelR ������ʵ��ǩ
% LabelH ����Ԥ���ǩ
if(nargin <3)
    Weight = ones(size(LabelR,1),1);
end
err = LabelH==LabelR;
ER = sum((Weight.*err)/sum(Weight));

    if(ER>0.5)
        ER = 0.5;
    end
    if(ER==0)
        ER=0.001;
    end
end