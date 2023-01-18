function Experiment()
    
    load(sprintf('data/%s', 'ROFFdata1.mat'));
    Src_data = OFFdata.Src_data;
    Tar_train = OFFdata.Tar_train;
    Tar_test = OFFdata.Tar_test;
    chunk_Max = length(Tar_train);
    Alldata = [];
    for i = 1:length(Tar_train)
        Alldata = [Alldata;Tar_train{i}];
    end
    % source_domains = {'amazon.mat','dslr.mat'};
    % target_domain = 'webcam.mat';
    % Experiment: the main function used to run HomOTL-ODDM.
    %
    %--------------------------------------------------------------------------
    % Input:
    %      source_domains: a set of source dataset_name, e.g. {'PIE1.mat','PIE2.mat','PIE3.mat','PIE4.mat'}
    %      target_domain: target dataset_name, e.g. 'PIE5.mat'
    %--------------------------------------------------------------------------
    
    %% run experiments:
    Options.theta = 0.001;
    Options.SVM_alpha = 0.001;
    Options.SVM_step = 100;
    Options.SVM_reg = 0.5;
    Options.Corp_update = 5;
    Options.Num_class = 10;
    
    resultICL = [];
    resultTL = [];
    resultCorp = [];
    AllSerr = [];
    AllTerr = zeros(chunk_Max,11);
    for run = 1:20
        run
        ht = [];
        oldcov = [];
        MS = [];
        for chunk_num = 1:chunk_Max
            chunk_num
            Cur_targetdata = Tar_train{chunk_num};
            Cur_test = Tar_test{chunk_num};
%             [Tar_AccICL,hts] = OnlineICL(Cur_targetdata,Cur_test,ht,Options);
%             [Tar_AccICLsep,ht] = OnlineICLsep(Cur_targetdata,Cur_test,ht,Options);
%             [Tar_AccTL,CORAL_Atl,ht,Serr,Terr] = OnlineTL(Cur_targetdata,Cur_test,Srct_data,Src_origX,hs,ht,Options,CORAL_Atl);
            [Tar_AccCorp,Tear_Acct,yPost_precs,Tear_Accs,ht,oldcov,MS] = OnlineCorp(Cur_targetdata,Cur_test,Src_data,ht,Options,oldcov,MS);
%             [Sear_Acc,Sear_Accs,Sear_Acct,NTear_Acc,Tear_Accs,Tear_Acct,CORAL_A,ht] = TestOnlineCorp(Cur_targetdata,Cur_test,Srct_data,Src_origX,hs,ht,Options,CORAL_A);
            resultCorp(run,chunk_num) = Tar_AccCorp;
            resultCorpy(run,chunk_num) = Tear_Acct;
            resultCorps(run,chunk_num) = yPost_precs;
            resultCorpt(run,chunk_num) = Tear_Accs;
%               resultTL(run,chunk_num) = Tar_AccTL;
%               resultTLt(run,chunk_num) = Terr;
%               resultTLs(run,chunk_num) = Serr;
%             resultICLsep(run,chunk_num) = Tar_AccICLsep;
%             resultICL(run,chunk_num) = Tar_AccICL;
%             AllTerr = [AllTerr;Terr];
%             AllSerr = [AllSerr;Serr];
%             AllTerr(chunk_num,1:length(Terr)) = Terr;

%         Sear(run,chunk_num) = Sear_Acc;
%         Sears(run,chunk_num) = Sear_Accs;
%         Seart(run,chunk_num) = Sear_Acct;
%         Tear(run,chunk_num) = NTear_Acc;
%         Tears(run,chunk_num) = Tear_Accs;
%         Teart(run,chunk_num) = Tear_Acct;

        end
%         save resultTL resultTL
%         save resultTLs resultTLs
%         save resultTLt resultTLt
%         save resultICLsep resultICLsep
%         save resultICL resultICL
                    save resultCorp resultCorp
          save resultCorpy resultCorpy
          save resultCorps resultCorps
          save resultCorpt resultCorpt
          save MS MS
S = mean((resultCorp(run,:)))
T = mean((resultCorpy(run,:)))
Tt = mean((resultCorpt(run,:)))

%           AVGTL = mean((resultTL(run,:)))
%           AVGTLs = mean((resultTLs(run,:)))
%           AVGTLt = mean((resultTLt(run,:)))
%           AVGCorp = mean((resultCorp(run,:)))
%         AVGICLsep = mean((resultICLsep(run,:)))
%         AVGICL = mean((resultICL(run,:)))

%     save Sear Sear
%     save Sears Sears
%     save Seart Seart
%     save Tear Tear
%     save Tears Tears
%     save Teart Teart

    end

