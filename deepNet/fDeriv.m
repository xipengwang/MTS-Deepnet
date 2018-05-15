function outputs = fDeriv(t,opt)
%opt = 1:linear
%opt = 2:LogSigmiod
%opt = 3:TanSigmoid
if(opt==1)
    outputs=ones(size(t));
elseif(opt==2)
    outputs=fSigmoid(t).*(1-fSigmoid(t));
else
    outputs = (-fTanSigmoid(t).^2) + 1;
end
end
function outputs=fSigmoid(t)
%
outputs = 1./(1+exp(-t));
end

function outputs=fTanSigmoid(t)
outputs = ((1+exp(t)) - (1+exp(-t)))./((1+exp(t)) + (1+exp(-t)));
end
