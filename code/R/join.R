
## merge the datasets
train_bodies <- read.csv("train_bodies.csv",header = TRUE,sep = ",",encoding = "UTF-8")
train_stances <- read.csv("train_stances.csv",header = TRUE,sep = ",",encoding = "UTF-8")
train<- merge(train_bodies,train_stances,by="Body.ID")
write.csv(train,file = "train.csv",row.names = FALSE)