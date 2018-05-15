function [data] = readData(path)
%path example:
%C:\Users\Xipeng1990\Desktop\Thesis\program\dataOutput\ID13_dev\dataSynchronizationOutput\

%data.positive: lane change samples
%column represents(1 - ):
%1          2           3           4           5           6           7
%Target     EcgRaw      HR          RR          GsrRaw      GSR         SCL
%8          9           10          11          12          13          14
%SCR        RspRaw      Exp Vol     Insp Vol    qDEEL       Resp Rate   Resp Rate Inst
%15         16          17          18          19          20          21
%Te         Ti          Ti/Te       Ti/Tt       Tpef/Te     Tt          Vent       
%22         23
%Vt/Ti      Work of Breathing
posK = 1;
negK = 1;
%%%%%debug
shift = 0;
%%%%%%debug
files=dir(path);
for i=3:length(files)
    tmpData = load([path files(i).name]);
    target = tmpData.syncTarget.data;
    posIdx = find(target(:,2)==1)-shift;
    for j=1:length(posIdx)/20
       dataIdx = posIdx((j-1)*20+1:j*20);
       record_targets = abs(target(dataIdx,2)-2);
       for k =0:19
           data.positive{posK}=[record_targets,...
               tmpData.syncECGraw.data(dataIdx-k,2),...
               tmpData.syncECG.data(dataIdx-k,2:3),...
               tmpData.syncGSRraw.data(dataIdx-k,2),...
               tmpData.syncGSR.data(dataIdx-k,2:4),...
               tmpData.syncRSPraw.data(dataIdx-k,2),...
               tmpData.syncRSP.data(dataIdx-k,2:15)];
           posK = posK + 1;
       end
    end
    negIdx = find(target(:,2)==2)-shift;
    for j=1:length(negIdx)/20
        dataIdx = negIdx((j-1)*20+1:j*20);
        record_targets = abs(target(dataIdx,2)-2);
        for k=0:19
             data.negative{negK}=[record_targets,...
               tmpData.syncECGraw.data(dataIdx-k,2),...
               tmpData.syncECG.data(dataIdx-k,2:3),...
               tmpData.syncGSRraw.data(dataIdx-k,2),...
               tmpData.syncGSR.data(dataIdx-k,2:4),...
               tmpData.syncRSPraw.data(dataIdx-k,2),...
               tmpData.syncRSP.data(dataIdx-k,2:15)];
           negK = negK + 1;
        end
    end
end

end

