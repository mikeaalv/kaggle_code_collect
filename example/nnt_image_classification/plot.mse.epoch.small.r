# plot mse through epoch
# train mse based on the first value among all the mse for trianing set
# test mse based on average value
rm(list=ls())
options(warn=1)
options(stringsAsFactors=FALSE)
options(digits=15)
require(stringr)
require(magrittr)
# require(R.matlab)
require(ggplot2)
require(reshape2)
# require(xml2)
# require(cowplot)
# require(scales)
# require(lhs)
# require(foreach)
# require(doMC)
# require(readr)
# require(abind)
projectpath="./"
infortab=read.table(file=paste0(projectpath,"submitlist.tab"),sep="\t",header=TRUE)
namevec=infortab[,"names"]
dirres=paste0(projectpath,"res/")
dirlist=1:28#c(1,2,3,4,8)#1:9#1:6
mselist=list(epoch=c(),train=c(),test=c(),names=c())
for(dir in dirlist){
  locdir=paste0(dirres,dir,"/")
  files=list.files(locdir)
  filesoutput=files[str_which(string=files,pattern="testmodel\\..+\\.out")]##might use testmodel\\..+\\.out for different filename (user should ensure)
  lines=readLines(paste0(locdir,filesoutput))
  train_ind=str_which(string=lines,pattern="Train Epoch:")
  test_ind=str_which(string=lines,pattern="test Loss")
  test_ind=test_ind[1:(length(test_ind)-1)]
  epoch_ind=sapply(seq(length(test_ind)),function(xi){
    if(xi>1){
      range=c(test_ind[xi-1],test_ind[xi])
    }else{
      range=c(0,test_ind[xi])
    }
    train_ind[train_ind<range[2]&train_ind>range[1]][1]
  })
  # trainrecord_ind=epoch_ind[seq(length(epoch_ind))%%2==0]
  lines[epoch_ind] %>% str_extract(string=.,pattern="Train Epoch:\\s+\\w+") %>%
              str_replace_all(string=,pattern="Train Epoch:\\s+",replacement="") %>%
              as.numeric(.) -> epoch_num
  lines[test_ind-1] %>% str_extract(string=.,pattern="Acc:\\s+[\\w\\.]+") %>%
              str_replace_all(string=,pattern="Acc:\\s+",replacement="") %>%
              as.numeric(.) -> acctrain
  lines[test_ind] %>% str_extract(string=.,pattern="Acc:\\s+[\\w\\.]+") %>%
              str_replace_all(string=,pattern="Acc:\\s+",replacement="") %>%
              as.numeric(.) -> acctest
  mselist[["epoch"]]=c(mselist[["epoch"]],epoch_num)
  mselist[["train"]]=c(mselist[["train"]],acctrain)
  mselist[["test"]]=c(mselist[["test"]],acctest)
  mselist[["names"]]=c(mselist[["names"]],rep(namevec[dir],times=length(epoch_num)))
}
msetab=as.data.frame(mselist)
colnames(msetab)=c("epoch","train","test","names")
msetablong=melt(msetab,id=c("epoch","names"))
msetablong[,"names"]=as.factor(msetablong[,"names"])
# msetablong[,"value"]=log10(msetablong[,"value"])
p<-ggplot(data=msetablong,aes(epoch,value,linetype=variable,colour=names))+
      geom_line(alpha=0.5)+
      scale_y_continuous(trans='log10',limits=c(0.4,max(msetablong[,"value"])))+
      xlab("epoch")+
      ylab("mse")+
      theme_bw()
ggsave(plot=p,file=paste0(dirres,"mse_epoch.pdf"))
##end state(epoch) statistics on mse
endwid=c(1,100)
names=unique(msetab[,"names"])
summtab=as.data.frame(matrix(NA,nrow=length(names),ncol=4))
colnames(summtab)=c("names","train_mean","test_mean","test_max")
rownames(summtab)=names
for(namegroup in names){
  subtab=msetab[msetab[,"names"]==namegroup,]
  endepoch=max(subtab[,"epoch"])
  subtab2=subtab[subtab[,"epoch"]>=(endepoch-endwid[2]),]
  summtab[namegroup,"names"]=namegroup
  summtab[namegroup,"train_mean"]=mean(subtab2[,"train"])
  summtab[namegroup,"test_mean"]=mean(subtab2[,"test"])
  summtab[namegroup,"test_max"]=mean(sort(subtab[,"test"],decreasing=TRUE)[endwid[1]])
}
save(summtab,msetablong,file=paste0(dirres,"Rplot_store.RData"))#,p
