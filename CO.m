function[Best_score,Best_pos,CO_curve,testY_best,test_target_grp_best,mae,mse,smape,r_squared]=CO(step_size,Camel_Caravan,Max_iterations,lowerbound,upperbound,dimension)
global length_test_target
 
Conv=zeros(1,Max_iterations);
Visibility = 0.1;
Tmin = 30;
Tmax = 60;

best_so_far = NaN(1, Max_iterations);
x_old = NaN(Camel_Caravan,dimension);


%% Initialization
for i=1:dimension
    x_old(:,i) = ceil((lowerbound(i) + rand(Camel_Caravan,1) .* (upperbound(i) - lowerbound(i))) / step_size(i)) * step_size(i);
end


fit = NaN(Camel_Caravan,1);
testY = cell(Camel_Caravan, 1);

for i = 1:Camel_Caravan
    testY{i} = NaN(length_test_target,1);
end

test_target_grp = cell(Camel_Caravan, 1);
for i = 1:Camel_Caravan
    test_target_grp{i} = NaN(1,length_test_target);
end

for i =1:Camel_Caravan
    L=x_old(i,:);
    [test_target_grp{i}, testY{i}, fit(i)]=myCostFunction(L(1), L(2), L(3));
end
[Fit_order, location]=sort(fit);
Fit_old = Fit_order(1);
x_old = x_old(location, :);
old_best = x_old(1,:);

disp(Fit_order);

test_target_grp = test_target_grp(location);
testY = testY(location);
test_target_grp_best = test_target_grp(1);
testY_best = testY(1);

v = rand(Camel_Caravan,1);

    %%
    lu=[lowerbound;upperbound];                         
    A=30;
    FES=0;
    C=0;
    Cmax=7;
    St=zeros(Cmax,dimension);
    B=200./(upperbound-lowerbound);
    
    %%
for t=1:Max_iterations
    T = unifrnd(Tmin,Tmax,Camel_Caravan,dimension);
    End = 1-(T-Tmin)/(Tmax-Tmin);
    for j=1:Camel_Caravan
        if v(j)>Visibility
            x(j,:) = x_old(j,:)+End(j,:).*(old_best-x_old(j,:));  
        else
            x(j,:) = (upperbound-lowerbound).*rand(1,dimension)+lowerbound;
        end
    end
    for i=1:Camel_Caravan
        x(i,:) = max(x(i,:), lowerbound);
        x(i,:) = min(x(i,:), upperbound);
    end
        
    for i =1:Camel_Caravan
    L=x(i,:);
    [test_target_grp{i}, testY{i}, fit(i)]=myCostFunction(L(1), L(2), L(3));
    end
    [Fit_order, location]=sort(fit);
    x_new = x(location, :);
    x_best = x_new(1,:);
    test_target_grp = test_target_grp(location);
    testY = testY(location);
    test_target_grp_new = test_target_grp(1);
    testY_new = testY(1);

    if Fit_order(1) < Fit_old
        old_best = x_best;
        Fit_old = Fit_order(1);
        test_target_grp_best = test_target_grp_new;
        testY_best = testY_new;
    end

        for i = 1:Camel_Caravan
        FES=FES+1;
        Conv(FES) = Fit_old;
        if FES>=Max_iterations
            break
        end

            %------------------- Guided Learning Strategy -----------------
        C=C+1;
        St(C,:)=x(i,:);

        if C>=Cmax
            V0=std(St,0,1).*B;
            C=0;
            FESold=FES;
            [old_best,Fit_old,x,fit,FES]=GLS(old_best,Fit_old,x,fit,lu,V0,A,FES,Max_iterations,i);
            Conv(FESold+1:FES) = Fit_old;
            if FES >= Max_iterations
                break
            end
        end
        end
    
     v = rand(Camel_Caravan,1);
     x_old = x;
     
    
     best_so_far(t)=Fit_old;
    CO_curve=best_so_far;
    Best_score=Fit_old;
    
    old_best(:,3) = ceil(old_best(:,3)); 
    Best_pos=old_best;

    
end
 
 test_target_grp_best = test_target_grp_best{1};
 testY_best = testY_best{1};
 
mae = mean(abs(test_target_grp_best - testY_best'));  % Mean Absolute Error
mse = mean((test_target_grp_best - testY_best').^2);  % Mean Squared Error
smape = mean(2 * abs(test_target_grp_best - testY_best') ./ (abs(test_target_grp_best) + abs(testY_best') + eps)) * 100; %symmetric mean absolute percentage error
% Calculate Coefficient of Determination (R^2)
ss_res = sum((test_target_grp_best - testY_best').^2);  % Residual sum of squares
ss_tot = sum((test_target_grp_best - mean(test_target_grp_best)).^2);  % Total sum of squares
r_squared = 1 - (ss_res / ss_tot);  % Coefficient of Determination (R^2)  

end