clc,clear all,close all
% Load data
in = xlsread('symptomOnsetHospitalization.xlsx')
age = in(:,3)
time = in(:,4)

% Descriptive statistics of whole datasets
descriptive=table()
descriptive.Mean=mean(time)'
descriptive.Min=min(time)'
descriptive.Max=max(time)'
descriptive.Std_dev=std(time)'
descriptive.Percentile_25=prctile(time,25)'
descriptive.Percentile_75=prctile(time,75)'


% group times by ages
T = table(age,time,'VariableNames',{'age','time'});
descriptive_age=table()
descriptive_age.group={}
descriptive_age.observations={}
descriptive_age.mean={}
descriptive_age.min={}
descriptive_age.max={}

groups={'0-9';'10-19';'20-29';'30-39';'40-49';'50-59';'60-69';'70-79';'80-89';'90+'};
for i = 1:length(groups)
    bin = T.time(10*(i-1) < T.age & T.age < 10*i-1);
    descriptive_age.group{i}=groups{i};
    descriptive_age.observations{i}=length(bin);
    descriptive_age.mean{i}=mean(bin);
    descriptive_age.min{i}=min(bin);
    descriptive_age.max{i}=max(bin);
end

% histogram of the dataset
figure(1)
histogram(time)
% distribution of the dataset
names={'Gamma','HalfNormal','exponential','logistic','rayleigh'};
pd = {};
h=[];
p=[];
st=[];
fit=table()

for i = 1:length(names)
    dist = names{i};
    fit.distribution{i}=dist;
    figure(i+1)
    pd{i} = fitdist(time,dist);
    [h(i),fit.p_value{i},stats] = chi2gof(time,'CDF',pd{i});
    fit.chi_sq{i}=stats.chi2stat;
    histfit(time,35,dist);
    title([dist ' distribution']);
    xlabel('days')
    ylabel('observations')
end