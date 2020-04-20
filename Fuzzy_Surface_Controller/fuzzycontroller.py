from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import math
import numpy as np

#Singleton values
CONST_VALUE_VECTOR = {'A': .05, 'B': .15, 'C':.3, 'D':.4, 'E': .5, 'F': .6}
#X triangle membership apexes (x axis intesection)
x_vec = [4, 7.5, 10.5, 14, 18]
#Y triangle membership apexes (y axis intesection)
y_vec = [4.5, 9, 14, 19]

x_memberships = []
y_memberships = []
value_matrix = []
#z function that for required_surface takes in x, y vals return all zs estimated
def z_function(x, y):
    z=[]
    e = math.e
    for i in range(len(x)):
        z_z = []
        for j in range(len(y)):
            z_z.append(0.6 * e ** (-0.003 * (x[i] - 20) ** 2 - 0.015 * (y[j]- 14) ** 2))
        z.append(z_z)
    return z

#plots a wireframe for x, y, z puts result in Results directory
def plot_wireframe(x = 0, y = 0, z = 0, title=""):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    if(z == 0):
        x = np.linspace(1, 20, 30)
        y = np.linspace(1, 20, 30)
        z = z_function(x, y)
    else:
        plt.gca().invert_yaxis()
    max = np.max(z)
    min = np.min(z)
    X, Y = np.meshgrid(x, y)
    Z = np.array(z)

    ax.plot_wireframe(X, Y, Z, color='green')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='winter', edgecolor='none')
    file_name = "Results/" + title + ".png"
    plt.savefig(file_name)
    plt.show()

#Gets closest value to one of the singletons set the key for the value vector (grid singleton values)
def get_closest_value(value):
    min_distance = float('inf')
    min_key = 'A'
    for key in CONST_VALUE_VECTOR:
        current_distance = abs(CONST_VALUE_VECTOR[key] - value)
        if(current_distance < min_distance):
            min_distance = current_distance
            min_key = key
    return min_key

#Builds the grid with singleton values for a set of x and y values (val vectors)
def find_z_estimates(x_vals, y_vals):
    actuals = z_function(x_vals, y_vals)
    grid = actuals
    for x in range(0,len(actuals)):
        for y in range(0,len(actuals[x])):
            grid[x][y] = get_closest_value(actuals[x][y])
    return grid

#Defines the triangular memberships given x or y vector
def define_memberships(val_vector):
    triangles = []
    for i in range(len(val_vector)):
        min_apex_max = [0, 0, 0]

        if(i > 0):
            min_apex_max[0] = (val_vector[i - 1] + .01)
        else:
            min_apex_max[0] = float('-inf')

        min_apex_max[1] = (val_vector[i])

        if(i < (len(val_vector) - 1)):
            min_apex_max[2] = (val_vector[i + 1] - .01)
        else:
            min_apex_max[2] = float('inf')
        triangles.append(min_apex_max)
    return triangles

#Gets the amount of belonging to each triangle for a given  value
def get_membership_amount(val, members, apexes):
    if(val <= apexes[0][1]):
        members.append(1)
        members.append(0)
        return members
    elif(val >= apexes[1][1]):
        members.append(0)
        members.append(1)
        return members
    length = apexes[1][1] - apexes[0][1]
    place = val - apexes[0][1]
    belonging = round((place/length),2)
    members.append(round((1 - belonging) , 2))
    members.append(round(belonging, 2))
    return members

#Gets the triangular membership(s) values for a given value
def get_membership(val, membership):
    min_apex_index = float('inf')
    max_apex_index = float('-inf')
    for i in range(len(membership)):
        if(membership[i][1] == val):
            max_apex_index = i
            min_apex_index = i
            break
        if(membership[i][1] > val):
            max_apex_index = i
            if(i > 0):
                min_apex_index = i-1
            else:
                min_apex_index = 0
            break
    if(max_apex_index < 0): #value wasn't found. Must be greater than last apex
        max_apex_index = (len(membership) -1)
        min_apex_index = (len(membership) -1)
    members = [min_apex_index, max_apex_index]
    apexes = [membership[min_apex_index], membership[max_apex_index]]
    return get_membership_amount(val, members, apexes)

#Calculates the output for a given point
def calculate_output_for_point(memberships, values):
    calculated_vals = {}
    keys = []
    mins = []
    #values are [y1[x1, x2, x3] y2[x1, x2, ] etc]
    #memberships are x1, x2
    mins.append(min(memberships[0][2], memberships[1][2]))
    mins.append(min(memberships[0][3], memberships[1][2]))
    mins.append(min(memberships[0][2], memberships[1][3]))
    mins.append(min(memberships[0][3], memberships[1][3]))
    keys.append(values[memberships[0][0]][memberships[1][0]])
    keys.append(values[memberships[0][1]][memberships[1][0]])
    keys.append(values[memberships[0][0]][memberships[1][1]])
    keys.append(values[memberships[0][1]][memberships[1][1]])

    for i in range(len(keys)):
        if keys[i] in calculated_vals:
            if(calculated_vals[keys[i]] < mins[i]):
                calculated_vals[keys[i]] = mins[i]
        else:
            calculated_vals[keys[i]] = mins[i]
    output = 0
    denominator = 0
    for key in calculated_vals:
        if(calculated_vals[key] > 0):
            output += (calculated_vals[key] * CONST_VALUE_VECTOR[key])
            denominator += calculated_vals[key]
    output /= denominator
    return output

#Gets estimation for a single (x, y) point
def get_individual_point_estimation(x, y):
    fuzzy_members = []
    fuzzy_members.append(get_membership(x, x_memberships))
    fuzzy_members.append(get_membership(y, y_memberships))
    output = calculate_output_for_point(fuzzy_members, value_matrix)
    return output

#Uses get_individual_point_estimation for a set of points and the produces an estimated surface
def estimate_surface():
    z = []
    x = np.linspace(1, 20, 30)
    y = np.linspace(1, 20, 30)
    for i in range(len(x)):
        z_z = []
        for j in range(len(y)):
            z_z.append(get_individual_point_estimation(x[i], y[j]))
        z.append(z_z)
    plot_wireframe(x, y, z, "estimation_surface")
    return x, y, z

#Gets the error between the required_surface and surface being estimated
def get_total_error(required_surface, produced_surface, x_for_err, y_for_err):
    total_points = 0
    total_error = 0
    z_errors = []
    for i in range(len(required_surface)):
        z_z_errors = []
        for j in range(len(required_surface[i])):
            total_points += 1
            current_error = math.sqrt(math.pow((required_surface[i][j] - produced_surface[i][j]),2))
            total_error += current_error
            z_z_errors.append(current_error)
        z_errors.append(z_z_errors)
    print("TOTAL ERROR: " + str(total_error))
    total_error /= total_points
    print("AVERAGE ERROR: " + str(total_error))
    plot_wireframe(x_for_err, y_for_err, z_errors, "SurfaceEstimationError")
    return total_error

x_memberships = define_memberships(x_vec)
y_memberships = define_memberships(y_vec)
value_matrix = find_z_estimates(x_vec, y_vec)
''' This can be used to set manual values of the grid. ** Must be same x/y length as x/y vectors
print(value_matrix)
value_matrix[0] = ['A','B','B','B']
value_matrix[1] = ['A','C','C','C']
value_matrix[2] = ['B', 'C', 'D', 'D']
value_matrix[3] = ['B', 'D', 'E', 'F']
value_matrix[4] = ['B', 'C', 'D', 'D']
print(value_matrix)'''

#Estimate points
print("Estimated value of (5, 5)")
print(get_individual_point_estimation(5, 5))
print("Estimated value of (16, 16)")
print(get_individual_point_estimation(16,16))

#Estimate surface get all values for error calculation - Plots estimated surface
x_for_err, y_for_err, z_estimate = estimate_surface()

#plots wireframe for required_surface
plot_wireframe(title="required_surface")

z_actual = z_function(x_for_err, y_for_err)
total_error = get_total_error(z_actual, z_estimate, x_for_err, y_for_err)
