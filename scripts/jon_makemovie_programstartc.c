
// whether to allow use of Python if desired
#define USEPYTHON (USINGPYTHON)


// whether to allow use of MPI if desired
#define USEMPI (USINGMPI)


#if(USINGPYTHON)
//http://wiki.python.org/moin/IntegratingPythonWithOtherLanguages
//http://docs.python.org/extending/embedding.html
// Python must come first since can modify standard headers
#include <Python.h>
#endif

static int runpy_interactive_system(int argc, char *argv[]);
static int runpy_interactive(int argc, char *argv[]);

int runpy_script_func(char *scriptname, char *funcname, int argsc, char *argsv[]);
// alt: using c inside python
//http://docs.python.org/extending/extending.html



#if(USEMPI)
#include <mpi.h>
#else
#define MPI_MAX_PROCESSOR_NAME 1000
#endif




#define MAXCHUNKS 100000
#define MAXCHUNKSTRING (MAXCHUNKS*(1+3)) // roughly 3 digits with 1 space for MAXCHUNKS numbers.  Only valid for log10(MAXCHUNKS)<~3
#define MAXGENNAME 2000

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <float.h>

#include <sys/stat.h> // for system unix commands.  See also "info libc"
#include <sys/types.h> // for system unix commands.  See also "man -a mkdir"
#include <unistd.h> // see man -a chdir






// C prototypes:
static int init_mpi(int *argc, char **argv[]);
static int myargs(int argc, char *argv[]);
static int get_chunklist(size_t strsize, char* chunkliststring, int *chunklist, int *numchunks);
static int print_chunklist(int numchunks,int *chunklist);
static int setup_tpy_makemovie(int myid, int *chunklist, int totalchunks, char *jobprefix, char *cwdold, char *cwdnew, int runtype);
static int finish_tpy_makemovie(int myid, int *chunklist, int totalchunks, char *jobprefix, char *cwdold, char *cwdnew);

static void cpu0fprintf(FILE* fileptr, char *format, ...);
static void myffprintf(FILE* fileptr, char *format, ...);


int truenumprocs,myid;
char processor_name[MPI_MAX_PROCESSOR_NAME];
int procnamelen;

int getchunklistfromfile;
int totalchunks;
char chunkliststring[MAXCHUNKSTRING];
int chunklist[MAXCHUNKSTRING];
int numchunks,numchunksactual;

char DATADIR[MAXGENNAME];
char jobprefix[MAXGENNAME];

char cwdold[MAXGENNAME];
char cwdnew[MAXGENNAME];





/////////////////////
//
// main C function
//
/////////////////////
int main(int argc, char *argv[])
{

  
  ///////////////////////////////////
  //
  myffprintf(stdout,"init_mpi Begin.\n");

  init_mpi(&argc,&argv);

  myffprintf(stdout,"init_mpi End: myid=%d.\n",myid);

  ///////////////////////////////////
  //
  myffprintf(stdout,"myargs Begin: myid=%d.\n",myid);

  int numargs;
  numargs=myargs(argc,argv);
  int system;
  system=atoi(*(argv+numargs+2));
  int parallel;
  parallel=atoi(*(argv+numargs+3));
  int runtype;
  runtype=atoi(*(argv+numargs+4));
  
  myffprintf(stdout,"myargs End: myid=%d. runtype=%d parallel=%d system=%d\n",myid,system,parallel,runtype);



  ///////////////////////////////////
  //
  //  (i.e. runchunkn.sh is called as the mpi binary in Jon's scripts.)
  myffprintf(stdout,"C runchunkn.sh -like Begin: myid=%d.\n",myid);

  int subjobnumber;
  subjobnumber=setup_tpy_makemovie(myid,chunklist,totalchunks,jobprefix,cwdold,cwdnew,runtype);

  myffprintf(stdout,"C runchunkn.sh -like End: myid=%d.\n",myid);


  ///////////////////////////////////
  //
  // Note that only myid==0's Fortran call will necessarily show Fortran code output.  Rest of CPU's output may not be redirected (i.e. that's implementation dependent)
  myffprintf(stdout,"C PY Begin: myid=%d.\n",myid);


  // TODOMARK: Here call for python script
  // kraken does have python2.7.1, but unsure about packages
  // should add module
  char scriptname[MAXGENNAME],funcname[MAXGENNAME];
  //  strcpy(scriptname,"~/py/mread/__init.py"); // no, can't assume home directory accessible (as issue on Kraken)
  strcpy(scriptname,"./py/mread/__init.py"); // assume full py directory copied to local run directory by makemovie.sh
  strcpy(funcname,"main");

  // go to directory where originally started job where python will be running from
  int error;
  error=chdir(cwdold);
  if(error!=0){
    myffprintf(stderr,"Failed to change to directory: %s\n",cwdold);
    exit(1);
  }

  //runpy_script_func(scriptname, funcname, argsc, argsv)
  // remove user args not meant for python
  
  if(argc-numargs>=2){
    fprintf(stderr,"argc-numargs=%d argv+numargs[1]=%s\n",argc-numargs,(argv+numargs)[1]); fflush(stderr);
  }
  else{
    fprintf(stderr,"No python args given\n"); fflush(stderr);
  }
  // fix-up argument for runi if correct runtype (all runtype's now)
  // argv[0]   argc=1  : python call
  // argv[1]   argc=2  : chunklisttype
  // argv[2]   argc=3  : chunklist
  // argv[3]   argc=4  : runn
  // argv[4]   argc=5  : DATADIR
  // argv[5]   argc=6  : jobcheck
  // argv[6]   argc=7  : init file
  // argv[7]   argc=8  : system (pytonarg=1)
  // argv[8]   argc=9  : parallel (pytonarg=2)
  // argv[9]   argc=10 : runtype (pytonarg=3)
  // argv[10]  argc=11 : modelname (pytonarg=4)
  // argv[11]  argc=12 : fakeruni (pythonarg=5)
  // argv[12]  argc=13 : runn (pythonarg=6)
  // argv[13+] argc=14 : <others> (pytonarg=7)
  if(argc-numargs>=5){ // at least runtype should be given
    if(1){
      // assumes large amount of space ready for this
      if(argc-numargs>=5){
        sprintf(*(argv+numargs+6),"%d",subjobnumber-1); // -1 because needs to be from 0 to n-1 while chunk# goes from 1 to n
      }
      else{
        fprintf(stderr,"No runi given"); fflush(stderr);
      }
    }
  }
  else{
    fprintf(stderr,"No runtype given"); fflush(stderr);
  }

  fprintf(stderr,"STEP2: %d\n",myid); fflush(stderr);
  
  ////////////////////////////////////////
  // before running python, copy over any .npy files needed.  Copy so for any system maximal distribution for disk system
  // E.g., on stampede, copy to /tmp/ and use that file.
  // to ensure no race condition by multiple cores, go through sequentially and test

#if(USEMPI)
      // Barrier required to ensure sequential access since no idea what order of cores is on system and how bunched into nodes.  Could use name of processor that id's the node, somewhow.
      MPI_Barrier(MPI_COMM_WORLD);
#endif


  // 1) Every core deletes the /tmp/*.npy files, can be done in race mode
  char systemstring[MAXCHUNKSTRING+MAXGENNAME];
  sprintf(systemstring,"rm -rf /tmp/*.npy");

  int id;
  int error1=0;
  for(id=0;id<truenumprocs;id++){
    error1=system(systemstring);
    
    if(error1==-1){
      fprintf(stderr,"AFork failed for command: %s",systemstring);
      exit(1);
    }
    if(error1>0){
      fprintf(stderr,"ACommand returned error1 for: %s",systemstring);
      exit(1);
    }      
  }
  fprintf(stderr,"STEP3: %d\n",myid); fflush(stderr);

#if(USEMPI)
      // Barrier required to ensure sequential access since no idea what order of cores is on system and how bunched into nodes.  Could use name of processor that id's the node, somewhow.
      MPI_Barrier(MPI_COMM_WORLD);
#endif

  /// get whether files exist for copying
  int fileexists1=0;
  if( access( "qty2.npy", F_OK ) != -1 ){
    // file exists
    fileexists1++;
  }
  else {
    // file doesn't exist
  }
  int fileexists2=0;
  if( access( "avg2d.npy", F_OK ) != -1 ){
    // file exists
    fileexists2++;
  }
  else {
    // file doesn't exist
  }

  fprintf(stderr,"STEP3b: %d %d\n",id,myid); fflush(stderr);
#if(USEMPI)
      // Barrier required to ensure sequential access since no idea what order of cores is on system and how bunched into nodes.  Could use name of processor that id's the node, somewhow.
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      fprintf(stderr,"STEP3c: %d %d\n",id,myid); fflush(stderr);

      fprintf(stderr,"STEP4: %d\n",myid); fflush(stderr);

  // 2) Every core that first sees no file, copies over the file.
  char systemstring1[MAXCHUNKSTRING+MAXGENNAME];
  char systemstring2[MAXCHUNKSTRING+MAXGENNAME];
  sprintf(systemstring1,"cp qty2.npy /tmp/");
  sprintf(systemstring2,"cp avg2d.npy /tmp/");

  int error2=0;
  for(id=0;id<truenumprocs;id++){
    if(myid==id){

      int fileexists1new=0;
      if( access( "/tmp/qty2.npy", F_OK ) != -1 ){
        // file exists
        fileexists1new++;
      }
      else {
        // file doesn't exist
      }
      int fileexists2new=0;
      if( access( "/tmp/avg2d.npy", F_OK ) != -1 ){
        // file exists
        fileexists2new++;
      }
      else {
        // file doesn't exist
      }

      if(fileexists1new==0 && fileexists1){

        error2=system(systemstring1);
          
        if(error2==-1){
          fprintf(stderr,"BFork failed for command: %s",systemstring1);
          exit(1);
        }
        if(error2>0){
          fprintf(stderr,"BCommand returned error2 for: %s",systemstring1);
          exit(1);
        }
      }
    

      if(fileexists2new==0 && fileexists2){
        
        error2=system(systemstring2);
        
        if(error2==-1){
          fprintf(stderr,"BFork failed for command: %s",systemstring2);
          exit(1);
        }
        if(error2>0){
          fprintf(stderr,"BCommand returned error2 for: %s",systemstring2);
          exit(1);
        }
      }

      fprintf(stderr,"STEP5a: %d %d\n",id,myid); fflush(stderr);

      
    }// end if myid==id

    fprintf(stderr,"STEP5b: %d %d\n",id,myid); fflush(stderr);
#if(USEMPI)
      // Barrier required to ensure sequential access since no idea what order of cores is on system and how bunched into nodes.  Could use name of processor that id's the node, somewhow.
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      fprintf(stderr,"STEP5c: %d %d\n",id,myid); fflush(stderr);
    
  }
  fprintf(stderr,"STEP6: %d\n",myid); fflush(stderr);
 

  ////////////////////////////////////////



  if(USINGPYTHON){
    runpy_interactive(argc-numargs, argv+numargs);
    // Need to exit (Ctrl-D EOF) and not quit(), or else entire code quits
  }
  else{
    // use system() or something perhaps calling python that way?
    runpy_interactive_system(argc-numargs, argv+numargs);
  }

  fprintf(stderr,"STEP7: %d\n",myid); fflush(stderr);

  // go back to directory where job list and completion info is kept
  error=chdir(cwdnew);
  if(error!=0){
    myffprintf(stderr,"Failed to change to directory: %s\n",cwdnew);
    exit(1);
  }


  myffprintf(stdout,"C PY Done: myid=%d.\n",myid);

  finish_tpy_makemovie(myid,chunklist,totalchunks,jobprefix,cwdold,cwdnew);


  myffprintf(stdout,"Done with jon_makemovie_programstart.c.\n");

#if(USEMPI)
  // finish up MPI
  // Barrier required
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
#endif

  fprintf(stderr, "END\n");
  fflush(stderr);
  exit(0);
  
  
  return(0);



}


// Note we pass pointers since MPI_init() modifies the values
static int init_mpi(int *argc, char **argv[])
{

#if(USEMPI)
  int ierr;
  ierr=MPI_Init(argc, argv);

  if(ierr!=0){
    myffprintf(stderr,"MPI Error during MPI_Init\n");
    exit(1);
  }
  
  MPI_Comm_size(MPI_COMM_WORLD, &truenumprocs); // WORLD total number of processors
  MPI_Comm_rank(MPI_COMM_WORLD, &myid); // WORLD proc id
  MPI_Get_processor_name(processor_name, &procnamelen); // to ensure really on certain nodes

  myffprintf(stderr, "WORLD proc: %d of %d on %s\n", myid,truenumprocs,processor_name);

  myffprintf(stderr, "end: init_MPI\n");
  fflush(stderr);
#else
  truenumprocs=1;
  myid=0;
#endif
  
  return(0);

}



static int myargs(int argc, char *argv[])
{
  int argi,numargs,numextraargs;
  size_t strsize;
  int i;

  numargs=1+4; // number of current-program (i.e. non-python) arguments

  ////////////////
  //
  // Get arguments from command-line
  //
  ////////////////
  if(argc<1+numargs){ // 1 is normal command-line argument
    // if argc!=1+numargs, then rest is for python
    myffprintf(stderr,"Not enough args given! argc=%d\n",argc);
    for(i=0;i<argc;i++){
      myffprintf(stderr,"argv[%d]=%s\n",i,argv[i]);
    }
    myffprintf(stderr,"Expected: <binaryname> 0 \"string of chunk numbers separated by spaces.\" <totalchunks> <DATADIR> <jobprefix>  <python script name and arguments>\n");
    myffprintf(stderr,"OR:\n");
    myffprintf(stderr,"Expected: <binaryname> 1 chunklistfile.txt <totalchunks> <DATADIR> <jobunique> <python script name and arguments>\n");
    myffprintf(stderr,"E.g.: mpirun -np 4 ./makemoviec 0 \"1 2 3 4\" 4 . pychunk123 <python script name and arguments>\n");
    exit(1);
  }
  else{
    // arguments should be same and in same order as for runchunkn.sh for easy calling in script that sets up the job
    // argci=0 would access command-line argument
    argi=1; // 1 is first true argument after command line and MPI extracted stuff
    // argi=1:
    myffprintf(stderr,"argv[%d]=%s\n",argi,argv[argi]);
    getchunklistfromfile=atoi(argv[argi]); argi++;
    if(getchunklistfromfile==0){
      // argi=2:
      myffprintf(stderr,"argv[%d]=%s\n",argi,argv[argi]);
      strcpy(chunkliststring,argv[argi]); argi++;
      strsize=strlen(chunkliststring);
      if(strsize>MAXCHUNKSTRING){
        myffprintf(stderr,"Increase MAXCHUNKSTRING or use malloc!\n");
        exit(1);
      }
      myffprintf(stderr,"strsize=%d chunkliststring=%s\n",(int)strsize,chunkliststring);
    }
    else{
      // argi=2:
      // Then 2nd argument is filename from where to get CHUNKLIST
      char chunkliststringfilename[MAXGENNAME];
      myffprintf(stderr,"argv[%d]=%s\n",argi,argv[argi]);
      strcpy(chunkliststringfilename,argv[argi]); argi++;
      myffprintf(stderr,"chunkliststringfilename=%s\n",chunkliststringfilename);
      FILE* chunklistfile;
      chunklistfile=fopen(chunkliststringfilename,"rt");
      if(chunklistfile==NULL){
        myffprintf(stderr,"Cannot open %s\n",chunkliststringfilename);
        exit(1);
      }
      else{
        // then create chunkliststring
        int index=0;
        char ch;
        while(!feof(chunklistfile)){
          ch=fgetc(chunklistfile);
          if(ch=='\n'){
            chunkliststring[index]='\0';
            break; // then done!
          }
          else{
            chunkliststring[index]=ch;
            index++;
          }
        }// end while if !feof()
        fclose(chunklistfile);
      }//end else if can open file

      // get string information like when chunklist on command line
      strsize=strlen(chunkliststring);
      if(strsize>MAXCHUNKSTRING){
        myffprintf(stderr,"Increase MAXCHUNKSTRING or use malloc!\n");
        exit(1);
      }
      myffprintf(stderr,"strsize=%d chunkliststring=%s\n",(int)strsize,chunkliststring);


    }// end else if reading in chunklist from a file
    // argi=3:
    myffprintf(stderr,"argv[%d]=%s\n",argi,argv[argi]);
    totalchunks = atoi(argv[argi]); argi++;
    // argi=4:
    myffprintf(stderr,"argv[%d]=%s\n",argi,argv[argi]);
    strcpy(DATADIR,argv[argi]); argi++;
    // argi=5:
    myffprintf(stderr,"argv[%d]=%s\n",argi,argv[argi]);
    strcpy(jobprefix,argv[argi]); argi++;



    int runtype;
    runtype=atoi(*(argv+numargs+4));
    fprintf(stderr,"runtype=%d\n",runtype); fflush(stderr);

    ///////////////////////
    //
    // now get chunk list
    //
    ///////////////////////
    get_chunklist(strsize,chunkliststring,chunklist,&numchunks);
    print_chunklist(numchunks,chunklist);


    if(runtype==2){
      int itemspergroup=atoi(*(argv+numargs+4+4));
      fprintf(stderr,"itemspergroup=%d\n",itemspergroup); fflush(stderr);

      numchunksactual=numchunks/itemspergroup;
      numchunksactual=numchunksactual+1;
    }
    else numchunksactual=numchunks;

    if(numchunksactual!=truenumprocs){
      myffprintf(stderr,"Must have numchunksactual=%d equal to truenumprocs=%d\n",numchunksactual,truenumprocs);
      myffprintf(stderr,"Required since cannot fork(), so each proc can only call 1 python call.\n");
      exit(1);
    }
    else{
      myffprintf(stderr,"Good chunk count: numchunksactual=%d\n",numchunksactual);
    }// end else if good numchunks
  }// end if good number of arguments


  return(numargs);

}

static int print_chunklist(int numchunks,int *chunklist)
{
  int i;

  myffprintf(stderr,"Got %d chunks\n",numchunks);
  myffprintf(stderr,"chunks are: ");
  for(i=0;i<numchunks;i++){
    myffprintf(stderr,"%d ",chunklist[i]);
  }
  myffprintf(stderr,"\n");

  return(0);
}


// fill chunklist[] with numbers from string
static int get_chunklist(size_t strsize, char* chunkliststring, int *chunklist, int *numchunks)
{
  char *nptr,*endptr;

  *numchunks=0; // initialize
  nptr=&chunkliststring[0]; // setup pointer to start reading string from

  while(1){
    chunklist[*numchunks]=(int)strtol(nptr,&endptr,10);

    if(strlen(nptr)==strlen(endptr)){
      // check if done with string
      // then not going anywhere anymore, so probably whitespace at end with no valid characters
      break;
    }
    else{

      if(chunklist[*numchunks]<=0){
        myffprintf(stderr,"Chunk number problem: %d\n",chunklist[*numchunks]);
        exit(1);
      }

      // iterate
      *numchunks = (*numchunks)+1;
      nptr=endptr;
      // DEBUG:
      //      myffprintf(stderr,"cl[%d]=%d %d %d : %s\n",*numchunks,chunklist[*numchunks-1],strlen(nptr),strlen(chunkliststring),nptr);
    }
  }

  return(0);
}




// do things like in runchunkn.sh script
static int setup_tpy_makemovie(int myid, int *chunklist, int totalchunks, char *jobprefix, char *cwdold, char *cwdnew, int runtype)
{
  int subchunk;
  int subjobnumber;
  char subjobname[MAXGENNAME];
  char subjobdir[MAXGENNAME];  
  int error;
  FILE *pychunkfile;


  /////////////////////
  //
  // get original working directory
  //
  /////////////////////
  int sizecwdold=MAXGENNAME;
  if(getcwd(cwdold,sizecwdold)==NULL){
    myffprintf(stderr,"Could not get old working directory.\n");
    exit(1);
  }
  else{
    fprintf(stderr,"cwdold=%s\n",cwdold); fflush(stderr);
  }


  // At the end, will need to wait for all processes to end.   Do so via a file for each myid.  Here we remove old file if it exists.
  // ensure to use same name when creating and checking in finish_tpy_makemovie()
  char finishname[MAXGENNAME];
  sprintf(finishname,"finish.%d",myid);
  remove(finishname);


  /////////////////////
  //
  // Change directory
  //
  /////////////////////
  // Chunk list should never be <1.  First chunk possible is 1 (i.e Fortran index type)
  subchunk=myid+1; // dose-out chunks by CPU id number.
  subjobnumber=chunklist[myid]; // access array with 0 as first element. (C index type.)
  //  sprintf(subjobname,"%sc%dtc%d",jobprefix,subjobnumber,totalchunks);
  sprintf(subjobname,"%s_%d",jobprefix,runtype); // all chunks in same directory
  sprintf(subjobdir,"%s/%s",DATADIR,subjobname);

  // assumes path exists.  Let fail if not, since then not setup properly using chunkbunch.sh script
  // new path for all future file accesses that don't give full path
  error=mkdir(subjobdir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  // don't exit in case just already made directory
  //if(error!=0){
  //  myffprintf(stderr,"Failed to make directory: %s\n",subjobdir);
  //  exit(1);
  // }

  error=chdir(subjobdir);

  if(error!=0){
    myffprintf(stderr,"Failed to change to directory: %s\n",subjobdir);
    exit(1);
  }

  ////////////////////////
  //
  // setup pychunk.dat
  //
  ////////////////////////
  char pychunkfilename[MAXGENNAME];
  sprintf(pychunkfilename,"pychunk.%s.%d.%d.dat",jobprefix,subjobnumber,totalchunks);
  pychunkfile=fopen(pychunkfilename,"wt"); // new file, do not append
  if(pychunkfile==NULL){
    myffprintf(stderr,"Failed to open pychunk.dat file\n");
    exit(1);
  }

  myffprintf(pychunkfile,"%d %d\n",subchunk,totalchunks);

  fclose(pychunkfile);

  ////////
  //
  // Remove old PY files
  //
  // remove() is just same as unlink() for files and rmdir() for directories, but directory must be empty to use remove() or rmdir() on directories
  //
  ///////
  //remove("python*.out");
  //  remove("");

  ////////////
  //
  // get new working directory
  //
  ///////////
  int sizecwdnew=MAXGENNAME;
  if(getcwd(cwdnew,sizecwdnew)==NULL){
    myffprintf(stderr,"Could not get new working directory.\n");
    exit(1);
  }


  // ready to call C PY code!


  return(subjobnumber);

}



// do things like in runchunkn.sh script AFTER binary is called
static int finish_tpy_makemovie(int myid, int *chunklist, int totalchunks, char *jobprefix, char *cwdold, char *cwdnew)
{
  char finishname[MAXGENNAME];
  FILE *myfinishfile;


  // change back to old working directory (this is where finish files will be located)
  //chdir(cwdold); // no, stay in jobprefix directory


  // first create my file since this CPU is done.
  sprintf(finishname,"finish.%d",myid);
  myfinishfile=fopen(finishname,"wt");
  if(myfinishfile==NULL){
    myffprintf(stderr,"Could not open %s file\n",finishname);
    exit(1);
  }
  myffprintf(myfinishfile,"1\n"); // stick a 1 in there so non-zero size
  fclose(myfinishfile);

  // now check if all other id's have created such a file
  int finished;
  while(1){

    finished=1; // guess that finished
    int i;
    for(i=0;i<numchunksactual;i++){
      sprintf(finishname,"finish.%d",i);
      myfinishfile=fopen(finishname,"rt");
      if(myfinishfile==NULL){
        finished=0;
        break; // no point in checking rest of files, so exit for loop
      }
      else{
        // then file exists, so no missing files so far
        fclose(myfinishfile);
      }
    }
    if(finished==0){
      // then pause before checking again
      sleep(60);
    }
    else{
      // then done!  So break
      break; // this exits the while(1) loop
    }

  }// end while(1)


  return(0);
}

// only myid==0 prints
static void cpu0fprintf(FILE* fileptr, char *format, ...)
{
  va_list arglist;

  if  (myid==0) {
    va_start (arglist, format);

    if(fileptr==NULL){
      fprintf(stderr,"tried to print to null file pointer: %s\n",format);
      fflush(stderr);
    }
    else{
      vfprintf (fileptr, format, arglist);
      fflush(fileptr);
    }
    va_end(arglist);
  }
}

// fprintf but also flushes
static void myffprintf(FILE* fileptr, char *format, ...)
{
  va_list arglist;

  va_start (arglist, format);

  if(fileptr==NULL){
    fprintf(stderr,"tried to print to null file pointer: %s\n",format);
    fflush(stderr);
  }
  else{
    vfprintf (fileptr, format, arglist);
    fflush(fileptr);
  }
  va_end(arglist);
}




static int runpy_interactive_system(int argc, char *argv[])
{
  fprintf(stdout,"Inside runpy_interactive_system: maxsize=%d\n",MAXCHUNKSTRING+MAXGENNAME); fflush(stdout);

  char systemstringtemp[MAXCHUNKSTRING+MAXGENNAME];
  char systemstring[MAXCHUNKSTRING+MAXGENNAME];

  fprintf(stdout,"Preparing systemstring: maxsize=%d\n",MAXCHUNKSTRING+MAXGENNAME); fflush(stdout);
  sprintf(systemstring,"python");
  int argi;
  for(argi=1;argi<argc;argi++){ // start at 1 since 0 contains garbage
    sprintf(systemstringtemp," %s %s",systemstring,argv[argi]);
    strcpy(systemstring,systemstringtemp);
  }
  // assume stdout and stderr outputted
  //  sprintf(systemstringtemp,"%s &> python.error",systemstring);
  //  strcpy(systemstring,systemstringtemp);

  fprintf(stdout,"(argc=%d) thing to run with system: %s\n",argc,systemstring); fflush(stdout);
  int error=0;
  error=system(systemstring);
  
  if(error==-1){
    fprintf(stderr,"Fork failed for command: %s",systemstring);
    exit(1);
  }
  if(error>0){
    fprintf(stderr,"Command returned error for: %s",systemstring);
    exit(1);
  }

  return 0;

}




//int PyRun_SimpleStringFlags(const char *command, PyCompilerFlags *flags)¶

// The Very High Level Layer
// http://docs.python.org/c-api/veryhigh.html#PyRun_SimpleFile
// Run script assuming main was passed exactly same arguments that get passed to python in makemovie.sh
static int runpy_interactive(int argc, char *argv[])
{

#if(USINGPYTHON)
  Py_Initialize();
  Py_Main(argc, argv);
  Py_Finalize();
#endif
  return 0;

}

//https://www6.software.ibm.com/developerworks/education/l-pythonscript/l-pythonscript-ltr.pdf
//http://www.ragestorm.net/tutorial?id=21#10
int runpy_script(int argsc, char *argsv[])
{
#if(USINGPYTHON)
  // Get a reference to the main module.
  PyObject* main_module = PyImport_AddModule("__main__");

  // Get the main module's dictionary
  // and make a copy of it.
  PyObject* main_dict = PyModule_GetDict(main_module);
  PyObject* main_dict_copy = PyDict_Copy(main_dict);

  // Execute two different files of
  // Python code in separate environments
  FILE* file_1 = fopen("file1.py", "r");
  PyRun_File(file_1, "file1.py", Py_file_input, main_dict, main_dict);

  //Py_Main(argc, argv);

  // Start Python Interpreter
  Py_Initialize();
  // Check if really loaded it
  if( !Py_IsInitialized() ) {
    printf("Unable to initialize Python interpreter.");
    return 1;
  }


  // End Python Interpreter
  Py_Finalize();

#endif
  return 0;

}

int runpy_code(char *code)
{

#if(USINGPYTHON)
  Py_Initialize();
  PyRun_SimpleString(code);
  Py_Finalize();
#endif
  return 0;

}

//http://justlinux.com/forum/showthread.php?t=151118
//PyCompilerFlags cf;
//cf.cf_flags = 0;
//PyRun_AnyFileFlags(stdin, "<stdin>", &cf);

// argsc and args[] start real args at args[0]
// assumes integer arguments for python function
int runpy_script_func(char *scriptname, char *funcname, int argsc, char *argsv[])
{
#if(USINGPYTHON)
  PyObject *pName, *pModule, *pDict, *pFunc;
  PyObject *pArgs, *pValue;
  int i;


  Py_Initialize();

  pName = PyString_FromString(scriptname);
  /* Error checking of pName left out */

  pModule = PyImport_Import(pName); // this imports the script
  Py_DECREF(pName);

  if (pModule != NULL) {
    pFunc = PyObject_GetAttrString(pModule, funcname);
    /* pFunc is a new reference */

    if (pFunc && PyCallable_Check(pFunc)) {
      pArgs = PyTuple_New(argsc);
      for (i = 0; i <argsc; ++i) {
        pValue = PyInt_FromLong(atoi(argsv[i]));
        if (!pValue) {
          Py_DECREF(pArgs);
          Py_DECREF(pModule);
          fprintf(stderr, "Cannot convert argument\n");
          return 1;
        }
        /* pValue reference stolen here: */
        PyTuple_SetItem(pArgs, i, pValue);
      }
      pValue = PyObject_CallObject(pFunc, pArgs); // this is where we call a function with arguments
      Py_DECREF(pArgs);
      if (pValue != NULL) {
        printf("Result of call: %ld\n", PyInt_AsLong(pValue));
        Py_DECREF(pValue);
      }
      else {
        Py_DECREF(pFunc);
        Py_DECREF(pModule);
        PyErr_Print();
        fprintf(stderr,"Call failed\n");
        return 1;
      }
    }
    else {
      if (PyErr_Occurred())
        PyErr_Print();
      fprintf(stderr, "Cannot find function \"%s\"\n", funcname);
    }
    Py_XDECREF(pFunc);
    Py_DECREF(pModule);
  }
  else {
    PyErr_Print();
    fprintf(stderr, "Failed to load \"%s\"\n", scriptname);
    return 1;
  }
  Py_Finalize();
#endif
  return 0;
}
