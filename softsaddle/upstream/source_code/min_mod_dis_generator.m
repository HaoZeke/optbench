function [displacement1, displacement2] = min_mod_dis_generator(v_min, sp_global_list, delta)
% generates two displacements along a given vector:
% displacement 1 has lenght delta
% displacement 2 has lenght delta as well but oriented in oposite direction 

displacement1 = sp_global_list + v_min * delta;
displacement2 = sp_global_list - v_min * delta;
