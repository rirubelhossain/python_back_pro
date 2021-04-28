import math as math_function
#neural network problem backwoard and forward problem 
#Here is the two input and two hidden layer and two output 
#b1 and b2 are bias 

#sigma function are define here 
def function_sigma(number ):
  number = (1+(math_function.exp(-number)))
  number = 1 / number ;
  return(number)


weight_mat_initial = [[0.15 , 0.20],
                     [0.25 , 0.30]]
second_mat = [[0.40 , 0.45],
              [0.50 , 0.55]]

second_mat_up = second_mat[0][0]

x1 = float(input("enter your main input such as x1"))#first input
x2 = float(input("enter your main input such as x2"))#second input 

b1 = float(input("enter your bias input such as b1"))#bias_1
b2 = float(input("enter your bias input such as b2"))#bias_2

h1 = ((x1 * weight_mat_initial[0][0]) + (x2 * weight_mat_initial[0][1]) + b1)
outh1 = function_sigma(h1)#first hidden layer function output that are contain outh1 


h2 = ((x1 * weight_mat_initial[1][0]) + (x2 * weight_mat_initial[1][1]) + b1)
outh2 = function_sigma(h2)#second hidden layer function output that are contain outh2

update_x1 = outh1
update_x2 = outh2

y1 = ((update_x1 * second_mat[0][0]) + (update_x2 * second_mat[0][1]) + b2)
y2 = ((update_x1 * second_mat[1][0]) + (update_x2 * second_mat[1][1]) + b2)

outy1 = function_sigma(y1)
outy2 = function_sigma(y2)

                                #now calculating error 
t1 = float(input("enter your target value t1"))
t2 = float(input("enter your target value t2"))

ita = float(input("enter learning rate "))

#define error function initail of the program 
result = 0.5*(math_function.pow((t1 - outy1),2))
result_1 = 0.5*(math_function.pow((t2 - outy2),2))

final_result = result + result_1 



#now update weights using back formula 
#error findout for w5 such as second_mat[0][0] indexing 

# first step dEtotal / dw5 

#we can write Detotal / Dw5 = (detotal / douty1) * ( douty1 / dy1) * ( dy1 / dw5)
                                    
                                    #w5_update 

detotal_douty1 = - (t1 - outy1)
up_hidden = detotal_douty1 ;

douty1_dy1 = outy1*( 1 - outy1)
up_hidden_outy1 = douty1_dy1 

dy1_dw5 = outh1

Detotal_Dw5 = detotal_douty1 * douty1_dy1 * dy1_dw5 # these equation is update for w5 which is second_mat[0][0]


second_mat[0][0] = second_mat[0][0] - ita * Detotal_Dw5 ; #second_mat[0][0] updated here 
print("the updated weight is w5 = ", second_mat[0][0])
                                      #w6_update

detotal_douty1 = - (t1 - outy1)

douty1_dy1 = outy1*( 1 - outy1)

dy1_dw5 = outh2


Detotal_Dw5 = detotal_douty1 * douty1_dy1 * dy1_dw5 # these equation is update for w5 which is second_mat[0][0]

second_mat[0][1] = second_mat[0][1] - ita * Detotal_Dw5 ; #second_mat[0][0] updated here 
print("the updated weight is w6 = ", second_mat[0][1])

                                      #w7_update

#we can write Detotal / Dw7 = (detotal / douty2) * ( douty2 / dy2) * ( dy2 / dw7)
detotal_douty1 = - (t2 - outy2)

douty1_dy1 = outy2*( 1 - outy2)
a_1 = detotal_douty1 ;
b_2 = douty1_dy1 ;

dy1_dw5 = outh1

Detotal_Dw5 = detotal_douty1 * douty1_dy1 * dy1_dw5

second_mat[1][0] = second_mat[1][0] - ita * Detotal_Dw5 ;
print("the updated weight is w7 = ", second_mat[1][0])

                                      #w8_update

#we can write Detotal / Dw7 = (detotal / douty2) * ( douty2 / dy2) * ( dy2 / dw7)
detotal_douty1 = - (t2 - outy2)

douty1_dy1 = outy2*( 1 - outy2)

dy1_dw5 = outh2

Detotal_Dw5 = detotal_douty1 * douty1_dy1 * dy1_dw5

second_mat[1][1] = second_mat[1][1] - ita * Detotal_Dw5 ;
print("the updated weight is w8 = ", second_mat[1][1])

                                        #hidden_layer_update_1

de1_douty1 = up_hidden ;
douty1_dy1_up = up_hidden_outy1 ;

final_hidden = de1_douty1 * douty1_dy1_up
de1_douth1_up = second_mat_up * final_hidden;


print("error_1_hidden_layer = ",de1_douth1_up) 


                                        #hidden_layer_update_2

de1_douty2 = a_1 ;
douty1_dy1_up_1 = b_2 ;
second_mat_up_2 = 0.50
final_hidden = de1_douty2 * douty1_dy1_up_1
de1_douth1_up_1 = second_mat_up_2 * final_hidden;

detotaL_final_douth1 =  de1_douth1_up + de1_douth1_up_1 

print("error_2_hidden_layer = ",de1_douth1_up_1)
print("final error = ", detotaL_final_douth1 ) 

up = outh1*(1- outh1);
print("final up the value = ", up );


dnet_h1_dw1 = x1 ;

print("the input value = ", dnet_h1_dw1);


#Putting it all together.
#finding total error respect with weight1 

detotal_dw1 = detotaL_final_douth1 * up * dnet_h1_dw1 ;
#print("the value of = ", detotal_dw1);


#We can now update w_1:

main_w1 =  weight_mat_initial[0][0] ;
update_w1 = (main_w1 - ita*detotal_dw1)#where ita is the learning rate 
main_w2 = weight_mat_initial[0][1] ;
update_w2 = (main_w2 - ita * detotal_dw1);
main_w3 = weight_mat_initial[1][0] ;
update_w3 = (main_w3 - ita *  detotal_dw1);
main_w4 = weight_mat_initial[1][1] ;
update_w4 = (main_w4 - ita * detotal_dw1) ;

print("the updated w1 = ", update_w1);
print("the updated w2 = ", update_w2);
print("the updated w3 = ", update_w3);
print("the updated w4 = ", update_w4);












