function [preY,h2]=N_TSK_Clustering(X,RuleNum,HI,Order,iniY,H1,H2,b1,b2,gamma)
%X:the given dataset,its size is N*d where N is the number of data points
%and d is the dimensionality
%RuleNum: number of fuzzy rules
%HI: number of fuzzy partitions
%Order: the order of a TSK fuzzy system, it is set to 0 for 0-TSK-FC and 1
%for 1-TSK-FC
%iniY: the initialized label vector of all data points in X, which can be
%obtained by FCM or k-means or other methods
%H1: number of iterations for convergence of Pgk
%H2: number of iterations for searching optimal performance
%b1 and b2: regularization parameters
%gamma: penalty parameter
[N,d]=size(X);    
[RuleBunch,~]=GetRuleBunch(RuleNum,HI,d,[]);
Delta=ones(RuleNum,d);   
labels=unique(iniY);
C=length(labels);
preY=iniY;
zt=TSK_RuleAvailable(X,RuleBunch,Delta,Order);
Pg=zeros(size(zt,2),C);
h2=1;
flag=0;
while h2<=H2 && flag==0
    for k=1:C
        Nk=zt(find(preY==labels(k)),:);
        Nrest=zt(find(preY~=labels(k)),:);
        centers(:,k)=mean(Nk,1)';
        NNk=Nk-repmat(centers(:,k)',size(Nk,1),1);
        Sk=NNk'*NNk;
        [V,D]=eig(Sk);
        [~,minPos]=min(diag(D));
        iniPgk=V(:,minPos);
        Pg(:,k)=calPgk(Nk,Nrest,iniPgk,b1,b2,gamma,H1);
    end
    for k=1:C
        dis(:,k)=(zt-repmat(centers(:,k)',N,1))*Pg(:,k);
    end
    [~,preY]=min(abs(dis),[],2);
    nmi(iniY,preY)
    if nmi(iniY,preY)>0.99
        flag=1;
    end
    iniY=preY;
    h2=h2+1;
end
end    

function zt=TSK_RuleAvailable(X,RuleBunch,Delta,Order)
zt=fromXtoZ_N_Order_1_c(X,RuleBunch,Delta,Order);
end

function [RuleBunch,RuleList]=GetRuleBunch(RuleNum,HI,D,RuleList)
    for i=1:RuleNum
        [~,RuleList]=ChooseRules(HI,D,RuleList);
    end
    RuleBunch=RuleList(end-RuleNum+1:end,:)/HI;
end

function [OneRule,RuleList]=ChooseRules(HI,D,RuleList)
    OneRuleX=zeros(1,D);
    while true
        for i=1:D
            OneRuleX(i)=randperm(HI+1,1)-1;
        end
        Repeated=HasBeenChoosed(OneRuleX,RuleList);
        if Repeated==false
            RuleList=[RuleList;OneRuleX];
            break;
        end
    end
    
    OneRule=OneRuleX/HI;
    clear OneRuleX
end

function [exist]=HasBeenChoosed(OneRule,RuleList)
    exist=false;
    
    if isempty(RuleList)
        return;
    end
    
    [rr,rc]=size(RuleList);
    for i=1:rr
        if length(OneRule(OneRule==RuleList(i,:)))==rc
            exist=true;
            break;
        end
    end
end

function Pgk=calPgk(Nk,Nrest,iniPgk,b1,b2,gamma,H)
    tol=0.001;
    tolt=1;
    h=1;
    Numk=size(Nk,1);
    ecol1=ones(Numk,1);
    ecol2=ones(size(Nrest,1),1);
    centerk=mean(Nk,1);
    NNk=Nk-repmat(centerk,size(Nk,1),1);    
    zt=[Nk;Nrest];
    while h<=H && tolt>tol  
        Sk0=NNk'*NNk;
        Sk1=diag(sign((Nrest-1/Numk*ecol2*ecol1'*Nk)*iniPgk));
        Sk2=sign(iniPgk'*(zt'*zt)*iniPgk-1);
        Sk3=(eye(size(zt,2))+b1*Sk0+gamma*Sk2*(zt'*zt))\(Nrest'-1/Numk*Nk'*ecol1*ecol2')*Sk1;
        HH=Sk1*(Nrest-1/Numk*ecol2*ecol1'*Nk)*Sk3;
        Pgk=Sk3*qpSOR((HH+HH')/2,0.7,b2,0.05);%by referring to the TWSVC KPPC in http://www.optimal-group.org/
        tolt=norm(Pgk-iniPgk);
        iniPgk=Pgk;
        h=h+1;
    end
end