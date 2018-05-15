function outputs = fActive(t,opt)
%opt = 1:linear
%opt = 2:LogSigmiod
%opt = 3:TanSigmoid
if(opt==1)
    outputs=fLinear(t);
elseif(opt==2)
    outputs=fSigmoid(t);
else
    outputs=fTanSigmoid(t);
end

end
function outputs=fSigmoid(t)
outputs = 1./(1+exp(-t));
end
function outputs=fLinear(t)
outputs = t;
end
function outputs=fTanSigmoid(t)
outputs = ((1+exp(t)) - (1+exp(-t)))./((1+exp(t)) + (1+exp(-t)));
end