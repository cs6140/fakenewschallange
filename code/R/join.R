
## merge the datasets
train_bodies <- read.csv("../../data/train_bodies.csv",header = TRUE,sep = ",",encoding = "UTF-8")
train_stances <- read.csv("../../data/train_stances.csv",header = TRUE,sep = ",",encoding = "UTF-8")
train<- merge(train_bodies,train_stances,by="Body.ID")
write.csv(train,file = "train1.csv",row.names = FALSE)
train <- read.csv("train.csv",header = T,sep = ",")

