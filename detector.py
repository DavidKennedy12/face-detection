import os
import cv2
import numpy as np
from glob import glob
import pickle

Num_of_images = 23708
size = 100
k = 11 #must always be an odd number, used in Nearest Neighbour matching
location = './UTKface/'  #file with the faces
files_list = sorted(glob(os.path.join(location, '*.jpg')))

#Used to unpack the cifar10 dataset
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic

#Used to store neighbours for clustering
class Object():
    def __init__(self, weights, isFace = False, age = 0): #default age = 0, for non faces
        self.age = age #age of the face, defaults to 0 for non faces
        self.isFace = isFace #whether an image is a fac
        self.weights = weights #the weights of the eigen face decomposition
    def get_distance(self, vector): #returns the euclidean distance between a given vector and this objects weight vector
        return np.linalg.norm(self.weights - vector)

#Used to sort the Neighbours in KNN
def getKey(item):
    return item[0]

#Find the K closest images from the training set
def KNN(list, vector):
    a = []
    for i in range(len(list)):
        dist = list[i].get_distance(vector)
        a.append([dist, list[i]])

    a = sorted(a, key = getKey) #sort the reference images by distance from the new image
    vote_true = 0 #number of neighbours that are faces
    vote_false = 0 #number of neighbours that aren't faces
    age = 0 #estimated age of the face, if applicable

    for j in range(k):
        if (a[j][1].isFace == True):
            vote_true += 1
            age += a[j][1].age #add the age of this face to the cumulative sum of the ages
        else:
            vote_false += 1

    if vote_true > vote_false:
        return True, age//vote_true #return average age of the nearby faces age
    else:
        return False, 0 #age = 0 if not a face

#returns true/flase if an image is a face, as well as the estimated age of the face
def classify_image(img, U, mean_face, list):
    img = np.reshape(img, (size**2,)) #ufold the array into a vector
    weights = U.T @ (img - mean_face) #obtain the weights of the eigen faces
    weights = weights / np.linalg.norm(weights) #Normalises the vector
    return KNN(list, weights)


def generate_eigen_faces():
    counter = 0
    face_vectors = np.zeros((1000, size**2)) #we will use 1000 face to compute the eigen faces, each vector is size*size in length
    j = 0
    for file in files_list:
        if(counter % 23 == 0): #Need this to ensure robust data, becuase the faces are sorted by age, sex, and ethnicity
            if (j == 1000):
                break
            img = cv2.imread(file, 0)
            img = cv2.resize(img, (size, size)) #Ensure all image are the same size
            face_vectors[j] = np.reshape(img, (size ** 2,)) #Turn into a vector
            j += 1
        counter += 1

    print("Calculating average face")
    mean_face = face_vectors.mean(axis=0)
    print("Computing the Eigen vectors and values")
    A = face_vectors - mean_face
    # Normally L = A.T @ A , but A is already transposed so in this case L = A @ A.T
    L = A @ A.T
    eigval, eigvec = np.linalg.eig(L)
    eigvec = A.T @ eigvec #get the eigen faces
    norms = np.linalg.norm(eigvec, axis=0)
    eigvec = eigvec / norms #normalise the eigen vectors, a.k.a Eigen Faces

    eigsum = np.sum(eigval) #sum of all eigen values
    csum = 0
    k90 = 0
    for i in range(len(eigvec)):
        csum = csum + eigval[i] #Cumulative sum of first "i" eigen values
        tv = csum / eigsum
        if tv > 0.90: #once we cover 90% of eigen values we can stop
            k90 = i
            break
    print("Done.", k90,"Eigen Faces span 90% of the basis")
    U = eigvec[:, 0:k90] #k90 is the number of eigen faces needed to cover 90% of the variation in images

    #save the eigen faces to file
    file_pickle = open('eigen_faces', 'wb')
    pickle.dump(U, file_pickle)
    #save the average face to a file
    file_pickle = open('mean_face', 'wb')
    pickle.dump(mean_face, file_pickle)
    return U, mean_face


def training(U, mean_face):
    print("Training non face images")
    list = []
    dic = unpickle("./cifar-10-batches-py/data_batch_1") #load cifar10 database
    for i in range(2000): #using 2000 non face images
        red_channel = dic[b'data'][i, 0:1024]
        green_channel = dic[b'data'][i, 1024:2048]
        blue_channel = dic[b'data'][i, 2048:3072]
        img = np.array([red_channel, green_channel, blue_channel])
        img = img.T
        img = np.reshape(img, (32, 32, 3))
        img = cv2.resize(img, (size, size)) #ensure consistent sizing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to black and white
        img = np.reshape(img, (size ** 2,))
        if i <= 255: #Completely monotone images, to help with categorizing non faces
            img = img*0 + i
        weights = U.T @ (img - mean_face)
        weights = weights / np.linalg.norm(weights)
        obj = Object(weights, False)
        list.append(obj)

    print("Training face images")
    counter = 0
    j = 0
    for file in files_list:
        counter += 1
        if j == 2000: #we will use 2000 face images
            break
        if counter % 11 == 0: #needed to get robust data, because faces are sorted by age, gender, and ethnicity
            img = cv2.imread(file, 0)
            img = cv2.resize(img, (size, size))
            img = np.reshape(img, (size ** 2,))
            weights = U.T @ (img - mean_face)
            weights = weights / np.linalg.norm(weights)
            age = int(file.split("/")[2].split("_")[0]) #the age of each person is the first number in the file name using undrscores as the delimiter
            obj = Object(weights, True, age)
            list.append(obj)
            j += 1

    print("Training Eigen Face weights saved")
    file_pickle = open('trained_list', 'wb')
    pickle.dump(list, file_pickle)
    return list


#takes in an a black and white image and represent it as a sum of the eigen faces
def reconstruct_image(img, U, mean_face):
    num_of_eigen_faces = len(U[0])
    img = cv2.resize(img, (size,size))
    cv2.imwrite('original_grey.png', img)
    img = np.reshape(img, (size**2,))
    a = U.T @ (img - mean_face)
    reconstructed = np.zeros(size**2)
    for i in range(num_of_eigen_faces):
        reconstructed += a[i] * U.T[i]

    reconstructed += abs(reconstructed.min()) #face may have negative value, so need to ensure smallest values = 0
    diff = (reconstructed.max() - reconstructed.min()) #the range of values
    reconstructed = reconstructed / (diff/255) #fit the value into the range of 0 - 255
    reconstructed = np.uint8(reconstructed)
    reconstructed = np.reshape(reconstructed, (size,size))
    cv2.imwrite("reconstructed_image.png", reconstructed)
    return


#Runs a size * size rectangle over the image at full resolution, half resolution and quarter resolution
#using differnt resolutions creates an image pyramid that can be used to detect faces at different scales
def multi_scale_recognition(img, U, mean_face):
    faces = 0 #count number of faces found
    height, width = img.shape
    original = np.array(img) #save for later
    face_start_position = []
    face_end_position = []

    # check if the entire image is a face
    test_patch = cv2.resize(original, (size, size))
    value, age = classify_image(test_patch, U, mean_face, list)
    if value == True: #i.e entier image is one face
        face_start_position.append([0, 0])
        face_end_position.append([width - 1, height - 1])
        cv2.rectangle(original, (0, 0), (width-1, height-1), (0, 0,0), 8)
        cv2.imwrite('multi_scale_face_recognition.png', original)
        return

    #Full resolution
    for i in range(height//size):
        for j in range(width//size):
            y_coord = i * size
            x_coord = j * size
            test_patch = img[y_coord:y_coord+size, x_coord:x_coord+size] #a size * size patch of the overall image to be checked for a face
            value, age = classify_image(test_patch, U, mean_face, list)
            if value is True:
                face_start_position.append([x_coord, y_coord]) #save the coordiantes of detected faces to be drawn as rectangles
                face_end_position.append([x_coord + size * 1, y_coord + size * 1])
                faces += 1

    #half resolution
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    height, width = img.shape
    for i in range(height//size):
        for j in range(width//size):
            y_coord = i * size
            x_coord = j * size
            test_patch = img[y_coord:y_coord+size, x_coord:x_coord+size]
            value, age = classify_image(test_patch, U, mean_face, list)
            if value is True:
                face_start_position.append([x_coord, y_coord])
                face_end_position.append([x_coord + size * 2, y_coord + size * 2]) #Need to scale end points as the rectangle will be drawn on original image not the scaled down image
                faces += 1

    #quarter resolution
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    height, width = img.shape
    for i in range(height // size):
        for j in range(width // size):
            y_coord = i * size
            x_coord = j * size
            test_patch = img[y_coord:y_coord + size, x_coord:x_coord + size]
            value, age = classify_image(test_patch, U, mean_face, list)
            if value is True:
                face_start_position.append([x_coord, y_coord])
                face_end_position.append([x_coord + size * 4, y_coord + size * 4]) #Quarter resolution so scale up by 4
                faces += 1

    #Draw the recognised faces as rectangles on the original image
    for i in range(faces):
        x_start= face_start_position[i][0]
        y_start= face_start_position[i][1]
        x_end = face_end_position[i][0]
        y_end = face_end_position[i][1]
        cv2.rectangle(original, (x_start, y_start), (x_end, y_end), (250, 250, 250), 4)

    cv2.imwrite('multi_scale_face_recognition.png', original)
    return


def test_classification_accuracy(U, mean_face, list):
    print("Testing classifier accuracy")
    correct = 0
    dic = unpickle("./cifar-10-batches-py/data_batch_2")
    for i in range(1000): #not faces
        red_channel = dic[b'data'][i, 0:1024]
        green_channel = dic[b'data'][i, 1024:2048]
        blue_channel = dic[b'data'][i, 2048:3072]
        img = np.array([red_channel, green_channel, blue_channel])
        img = img.T
        img = np.reshape(img, (32, 32, 3)) #convert into a 32*32 image with 3 colour channels
        img = cv2.resize(img, (size, size))  # ensure consistent sizing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to black and white
        img = np.reshape(img, (size ** 2,))
        value, age = classify_image(img, U, mean_face, list)
        if value == False:
            correct += 1
    print("Of 1000 non face images,", correct, "were correctly categorized")
    counter = 0

    j = 0
    correct = 0
    for file in files_list: #faces
        counter += 1
        if j == 1000:
            break
        if counter % 19 == 0: #selects a wide range of images, since they are sorted by age
            img = cv2.imread(file, 0)
            img = cv2.resize(img, (size, size))
            value, age = classify_image(img, U, mean_face, list)
            if value == True:
                correct += 1
            j += 1
    print("Of 1000 face images,", correct, "were correctly categorized")
    return


def test_age_accuracy(U, mean_face, list):
    print("Testing age classifier")
    correct = 0
    counter = 0
    j = 0
    error = 0
    for file in sorted(files_list):
        counter += 1
        if j == 1000:
            break
        if counter % 19 == 0: #selects a wide range of images, since they are sorted by age
            real_age = int(file.split("/")[2].split("_")[0])
            img = cv2.imread(file, 0)
            img = cv2.resize(img, (size, size))
            value, predicted_age = classify_image(img, U, mean_face, list)
            if value == True:
                j += 1
                correct += 1
                print(j, "(real age, estimated): ", real_age, predicted_age,)
                error += np.linalg.norm(real_age-predicted_age)
    print("Average difference between actual age and predicted age: ", error/correct)
    return 0


# ------------------ START HERE ------------------

# Train a new recognition model:
#Calculate the average face, and the eigen faces, U = matrix with the eigen vectors as columns
# U, mean_face = generate_eigen_faces()
#train a K Nearest Neighbours classifier using both faces and non-faces
# list = training(U, mean_face)

# Use pre-trained data files:
#load the eigen face data from a file
file_faces = open('eigen_faces', 'rb')
U = pickle.load(file_faces) #U = a matrix with the eigen faces as columns
#load the average face data from a file
file_mean_face = open('mean_face', 'rb')
mean_face = pickle.load(file_mean_face)
#load the pre-trained images from a file
file_list = open('trained_list', 'rb')
list = pickle.load(file_list)

#Load a file to apply multi scale recognition to
print("Multi Scale Face Matching Running")
im = cv2.imread("./Solvay_conference_1927.jpg", 0)
multi_scale_recognition(im, U, mean_face)
print("Multi Scale Face Matching Done")


# TESTING:

# Save the mean face as an image to a file
# mean_face_image = np.uint8(mean_face)
# mean_face_image = np.reshape(mean_face_image, (size,size))
# cv2.imwrite('mean_face.png', mean_face_image)

# Save an eigen the first eigen face to a file
# eigen_face_1 = U.T[0]
# diff = (eigen_face_1.max() - eigen_face_1.min())
# eigen_face_1 = eigen_face_1 / (diff/255)
# eigen_face_1 = np.uint8(eigen_face_1)
# eigen_face_1 = np.reshape(eigen_face_1, (size,size))
# cv2.imwrite('eigen_face_1.png', eigen_face_1)
# im = cv2.imread("./original_face.png", 0)
# reconstruct_image(im, U, mean_face)
# test_classification_accuracy(U, mean_face, list)
# test_age_accuracy(U, mean_face, list)