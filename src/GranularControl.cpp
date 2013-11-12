#include "testApp.h"


const int Nx = 352;
const int Ny = 288;
// A vector with all the frames.
vector <cv::Mat> TheFramesInput;

vector<ofVec3f> FirstGrains;
vector<ofVec3f> SecondGrains;
vector<ofVec3f> OutGrains;
vector<ofVec3f> GrainMask;
vector<float> randAng;


const int Nlast = 400;
ofVec3f BlockDims;
float inOp;//  overlaping porcentage
float outOp;//  overlaping porcentage
float NewoutOp;
int Gs = 51; // for fixed grain size in all dimensions
cv::Mat CurrentOutput;
int fc=0; // frame counter


void InitGrains(void);

float CalculateScalecorrect(void);
float ScaleCorrectFactor=1.0;


void GrainRandomizer();

int circuIndex = Nlast-1;


// Control global variables:
float SLD1 = 0;
float SLD2 = 0;
float SLD3 = 0;
float SLD4 = 0;
float SLD5 = 1.0;
float SLD6 = 0.0;
float SLD7 = 0.0;
bool boolFixedDelay = true;
bool boolRotGrains = false;
bool boolDelaygrains = false;
bool UpdateVar = true; // variable to update all the info from control and interface.

bool boolhueshift = false;
bool boolSatChange = false;
bool boolGeomGrains = false;
bool boolShiftGrain = false;
bool boolCam2Grain = false;

// activates change detection
bool boolChangeDet = false;

// activates face tracking
bool boolFace = false;

cv::Mat Cam2Frame;

// Audio

int BuffS = 2048;

// Face detect variables

cv::CascadeClassifier face_cascade;
cv::Rect FirstFace;

// audio based
bool boolAudioCircle = false;
bool boolAudioRandom = true;
vector<int> GrainsRandom;



//--------------------------------------------------------------
void testApp::setup(){
   
    // First creating an empty array
    
    BlockDims.x = Nx;
    BlockDims.y = Ny;
    BlockDims.z = Nlast;
    cv::Mat AuxMat(Ny,Nx,CV_8UC3,cv::Scalar(0,0,0));
    for (int k = 0; k< Nlast; k++){
        TheFramesInput.push_back(AuxMat.clone());
    }
    
    
   // vidGrabber.setVerbose(true);
 //   vidGrabber.initGrabber(Nx,Ny);
    
  //  ofSetFrameRate(15);
    inOp =50;
    outOp =50;
    NewoutOp = outOp;

    InitGrains();
    

    
    
    
//    UpdateSchedulerUpDown(TheUf,1);
    ofSetLogLevel(OF_LOG_VERBOSE);
    vidGrabber1.setDeviceID(1);
    vidGrabber1.initGrabber(Nx,Ny);
    vidGrabber2.setDeviceID(0);
    vidGrabber2.initGrabber(Nx,Ny);
    
    //vidGrabber.initGrabber(Nx,Ny);

//    std::exit(1);
    colorImg.allocate(Nx,Ny);
	grayImage.allocate(Nx,Ny);
    
    
    // Midi
    
    // open port by number
	midiIn.openPort(0);
	//midiIn.openPort("IAC Pure Data In");	// by name
	//midiIn.openVirtualPort("ofxMidiIn Input");	// open a virtual port
	
	// don't ignore sysex, timing, & active sense messages,
	// these are ignored by default
	midiIn.ignoreTypes(false, false, false);
	
	// add testApp as a listener
	midiIn.addListener(this);
	
	// print received messages to the console
	midiIn.setVerbose(false);
    
    
    
    // scale correction  factor 50% overlap
    ScaleCorrectFactor =CalculateScalecorrect();
    
    
    AudioValues.assign(BuffS, 0.0);
    
    
    cout<<"aca van: "<<endl;
    soundStream.listDevices();
    
    // selecting the microhpone
   // soundStream.setup(this, 0, 2, 44100, BuffS, 4);
    
    //selecting the logitech camera mic
    soundStream.setDeviceID(2);
 
    soundStream.setup(this, 0, 1, 48000, BuffS, 4);
    
    
    // load cascade detector files
    
    //-- 1. Load the cascades
    if( !face_cascade.load( "/Users/javiervillegas/of_v0.7.4_osx_release/apps/myApps/GranAutomatic2/haarcascade_frontalface_alt_tree.xml" ) ){
        // if( !face_cascade.load( "/Users/javiervillegas/of_v0.7.4_osx_release/apps/myApps/FaceCenter/haarcascade_mcs_eyepair_big.xml" ) ){
        //  if( !face_cascade.load( "/Users/javiervillegas/of_v0.7.4_osx_release/apps/myApps/FaceCenter/haarcascade_mcs_nose.xml" ) ){
        
        cout<<"--(!)Error loading cara \n"<<endl;
    };
    
    
    int Osamp = (int)(ceil(Gs*inOp/100.0));
    int jp = Gs - Osamp;
    
    
    
    // fixed output size
    int outOsamp = (int)(ceil(Gs*outOp/100.0));
    int ojp = Gs - outOsamp;
    int outYsize = Gs + ojp*((BlockDims.y-Gs)/jp);
    int outXsize = Gs + ojp*((BlockDims.x-Gs)/jp);
    
    // creating an empty frame for the output
    
    CurrentOutput =cv::Mat::zeros(outYsize,outXsize, CV_8UC3);
    
    
    
}

//--------------------------------------------------------------

void testApp::update(){
    bool bNewFrame = false;
    

    vidGrabber1.update();

        
    bNewFrame = vidGrabber1.isFrameNew();
   
    //
    if (bNewFrame){
        //
        // circular buffer

        colorImg.setFromPixels(vidGrabber1.getPixels(), Nx,Ny);
          
        cv::Mat AuxMat;
        AuxMat = colorImg.getCvImage();
        AuxMat.copyTo(TheFramesInput[circuIndex]);
        if (boolFace){
            detectFace(AuxMat);
            
            for (int g=0; g < GrainMask.size(); g++) {
                
                if((GrainMask[g].x+Gs/2.0>FirstFace.x)&&(GrainMask[g].y+Gs/2.0>FirstFace.y)&&
                   (GrainMask[g].x+Gs/2.0<(FirstFace.x + FirstFace.width))&&
                   (GrainMask[g].y+Gs/2.0<(FirstFace.y + FirstFace.height)))
                     {
                         GrainMask[g].z = 1.0;
                     }
                else{
                
                    GrainMask[g].z = 0.0;
                
                }
            }
                UpdateVar = true;
       }
        
        
        
        fc = circuIndex;
        // Fc has the current sample index
        circuIndex--;
        if (circuIndex<0){
            circuIndex = Nlast-1;
        }

        
        
        // Motion Detection
        
        if(boolChangeDet){
            detectMotion(2);
        }
        
        if(boolAudioCircle){
            EnergyCenterSelect();
        }
        
        if(boolAudioRandom){
            AudioRandomSelect();
        }
        
        // Getting a frame from the other camera
        vidGrabber2.update();
       // if (vidGrabber2.isFrameNew()){
        colorImg2.setFromPixels(vidGrabber2.getPixels(),Nx,Ny);
        Cam2Frame = colorImg2.getCvImage();
     //   }
        
        
        
        
        
        
     // In case update is needed:
        if(UpdateVar){
        
            
            if (boolDelaygrains){
            
                for (int g =0; g < SecondGrains.size(); g++) {
                    if(GrainMask[g].z ==1.0){
                        if (boolFixedDelay){
                            SecondGrains[g].z = (int)((SLD2/127.0)*(Nlast/2.0));
                        
                        }
                        else{
                            SecondGrains[g].z = (int)(ofRandom((SLD2/127.0)*(Nlast/2.0)));
                        }
                    }
                    else{
                        SecondGrains[g].z = 0.0;
                    }
                }
            
            
            }
        
        
        
        
            UpdateVar = false;
        }
        
        
        
        
        
        
        
    // calculate new output size:
    int outOsamp = (int)(ceil(Gs*outOp/100.0));
    int ojp = Gs - outOsamp;
 
    // input jump size
    int Osamp = (int)(ceil(Gs*inOp/100.0));
    int jp = Gs - Osamp;

    
    int outYsize = Gs + ojp*((BlockDims.y-Gs)/jp);
    int outXsize = Gs + ojp*((BlockDims.x-Gs)/jp);
    
   // creating an empty frame for the output

    CurrentOutput =cv::Mat::zeros(outYsize,outXsize, CV_8UC3);
 
    // pointer to output data:
    
    unsigned char *output = (unsigned char*)(CurrentOutput.data);
    
    // running throught the list of grains
    
    for (int g =0; g < FirstGrains.size(); g++) {
        
        
        // two halves
        for (int m=0; m<2; m++) {
          // copy the grain from the input
            int IntiFrame;
            if(m==1){
                    IntiFrame = (fc + (int)FirstGrains[g].z)%Nlast;
             }
            else{
                    IntiFrame = ((fc + (int)SecondGrains[g].z)%Nlast);
            }
            
            //int IntiFrame = (m==1)?(fc + (int)FirstGrains[g].z)%Nlast:((fc + (int)SecondGrains[g].z)%Nlast);
            
            unsigned char *input = (unsigned char*)(TheFramesInput[IntiFrame].data);
            unsigned char *input2 = (unsigned char*)(Cam2Frame.data);
            
            for (int x =0; x<Gs; x++) {
                for (int y=0; y<Gs; y++) {
                    
                    float rIn,gIn,bIn;
                    float rOut,gOut,bOut;
                    float Scale;
                    Scale = ((0.5*0.5*0.5)*(1.0 -cosf(2*PI*x/(float)(Gs-1)))*
                             (1.0 -cosf(2*PI*y/(float)(Gs-1)))*
                             (1.0 -cosf(2*PI*(m*(Gs-1)/2.0 +(fc%jp))/(float)(Gs-1))));
                    // m =0 is the next grain
                    
                    Scale/=(ScaleCorrectFactor*ScaleCorrectFactor*ScaleCorrectFactor);
                    // pixels at the inut
                    int yindIn;
                    int xindIn;
                    
                    if ((!boolRotGrains)||(GrainMask[g].z==0.0)) {
                    
                        if (m==1) {
                            xindIn = x + FirstGrains[g].x;
                            yindIn = y + FirstGrains[g].y;
                        }
                        else {
                            xindIn = x + SecondGrains[g].x;
                            yindIn = y + SecondGrains[g].y;
                        }
                    }
                    if ((boolRotGrains)&&(GrainMask[g].z==1.0)){
                    
                        if (m==1) {
                            float newX = (x + FirstGrains[g].x) - (FirstGrains[g].x+Gs/2.0);
                            float newY = (y + FirstGrains[g].y) - (FirstGrains[g].y+Gs/2.0);
                            xindIn = (int)(newX*cos(PI*SLD1/127.0) - newY*sin(PI*SLD1/127.0)) + (FirstGrains[g].x+Gs/2.0);
                            yindIn = (int)(newX*sin(PI*SLD1/127.0) + newY*cos(PI*SLD1/127.0)) + (FirstGrains[g].y+Gs/2.0);
                        }
                        else {
                            float newX = (x + SecondGrains[g].x) - (SecondGrains[g].x+Gs/2.0);
                            float newY = (y + SecondGrains[g].y) - (SecondGrains[g].y+Gs/2.0);
                            xindIn = (int)(newX*cos(PI*SLD1/127.0) - newY*sin(PI*SLD1/127.0)) + (SecondGrains[g].x+Gs/2.0);
                            yindIn = (int)(newX*sin(PI*SLD1/127.0) + newY*cos(PI*SLD1/127.0)) + (SecondGrains[g].y+Gs/2.0);

                        }
                    
                    
                    }
                    
                    
                    // Geometrical transformations:
                    // center
                    
                    if ((boolGeomGrains)&&(GrainMask[g].z==1.0)){
                        
                        float newX,newY;
                        if (m==1) {
                            newX = (xindIn - FirstGrains[g].x - Gs/2.0);
                            newY = (yindIn - FirstGrains[g].y - Gs/2.0);
                        }
                        
                        else{
                            newX = (xindIn - SecondGrains[g].x - Gs/2.0);
                            newY = (yindIn - SecondGrains[g].y - Gs/2.0);
                        }

                        // First converting to polar and normalizing magnitude
                    
                        float Rf = sqrtf(2*(newX*newX+newY*newY))/Gs;
                        float newR = powf(Rf, 0.6+2*SLD5/127.0);
                    
                        newY = (Rf!=0)?(newR*newY/Rf):0.0;
                        newX = (Rf!=0)?(newR*newX/Rf):0.0;
                    
                        if (m==1) {
                            xindIn = (int)(newX) + (FirstGrains[g].x+Gs/2.0);
                            yindIn = (int)(newY) + (FirstGrains[g].y+Gs/2.0);
                        }
                    
                        else{
                            xindIn = (int)(newX) + (SecondGrains[g].x+Gs/2.0);
                            yindIn = (int)(newY) + (SecondGrains[g].y+Gs/2.0);
                        }
                    
                    }
                    
  
                    
                    xindIn = (xindIn<0)?0:xindIn;
                    xindIn = (xindIn>outXsize-1)?BlockDims.x:xindIn;
                    
                    yindIn = (yindIn<0)?0:yindIn;
                    yindIn = (yindIn>outYsize-1)?BlockDims.y:yindIn;
                    
                    
                    
                    
                    
                    if((boolCam2Grain)&&(GrainMask[g].z==1.0)){
                        bIn = (float)input2[(int)(3*BlockDims.x * (yindIn) + 3*(xindIn)) ] ;
                        gIn = (float)input2[(int)(3*BlockDims.x * (yindIn) + 3*(xindIn) + 1)];
                        rIn = (float)input2[(int)(3*BlockDims.x * (yindIn) + 3*(xindIn) + 2)];
                    }
                    else{
                        bIn = (float)input[(int)(3*BlockDims.x * (yindIn) + 3*(xindIn)) ] ;
                        gIn = (float)input[(int)(3*BlockDims.x * (yindIn) + 3*(xindIn) + 1)];
                        rIn = (float)input[(int)(3*BlockDims.x * (yindIn) + 3*(xindIn) + 2)];
                    
                    }

                    
                    
                    // Hue and saturation transformations:
                    
                    
                    float Ang;
                    if(boolhueshift&&(GrainMask[g].z==1.0)){
                        Ang = SLD3/127.0*2*PI;
                    }
                    else{
                        Ang =0;
                    }
               
                    float S;
                    if(boolSatChange&&(GrainMask[g].z==1.0)){
                      S = 2.1*SLD4/127.0;
                    }
                    else{
                        S = 1.0;
                    }
                    float SU = S*cos(Ang);
                    float SW = S*sin(Ang);
                    float rMed,gMed,bMed;
                    
                  
                    rMed = (.299+.701*SU+.168*SW)*rIn
                    + (.587-.587*SU+.330*SW)*gIn
                    + (.114-.114*SU-.497*SW)*bIn;
                    gMed = (.299-.299*SU-.328*SW)*rIn
                    + (.587+.413*SU+.035*SW)*gIn
                    + (.114-.114*SU+.292*SW)*bIn;
                    bMed = (.299-.3*SU+1.25*SW)*rIn
                    + (.587-.588*SU-1.05*SW)*gIn
                    + (.114+.886*SU-.203*SW)*bIn;
                   
                    
                    // from:
                    // http://beesbuzz.biz/code/hsv_color_transforms.php
                    
                    
                    
                    
                    
                    
                    
                    //pixels at the output
                    // output indexes
                    
                    int xindOut;
                    int yindOut;
                    xindOut = x + OutGrains[g].x ;
                    yindOut = y + OutGrains[g].y;
                    if((boolShiftGrain)&&(GrainMask[g].z==1.0)){
                        xindOut = x + OutGrains[g].x + CurrentEnergy*5.0*SLD6/127.0*Gs*cos(randAng[g]);
                        yindOut = y + OutGrains[g].y + CurrentEnergy*5.0*SLD6/127.0*Gs*sin(randAng[g]);
                    }

                  
                    xindOut = (xindOut<0)?0:xindOut;
                    xindOut = (xindOut>outXsize-1)?outXsize-1:xindOut;
                    
                    yindOut = (yindOut<0)?0:yindOut;
                    yindOut = (yindOut>outYsize-1)?outYsize-1:yindOut;
                    
                    
                    
                    bOut = (float)output[3*CurrentOutput.cols * (yindOut) + 3*(xindOut) ] ;
                    gOut = (float)output[3*CurrentOutput.cols * (yindOut) + 3*(xindOut) + 1];
                    rOut = (float)output[3*CurrentOutput.cols * (yindOut) + 3*(xindOut) + 2];
                    

                    output[3*CurrentOutput.cols * (yindOut) + 3*(xindOut) ] = (uchar)(bOut + Scale*(1.0 - boolShiftGrain*.5)*bMed);
                    output[3*CurrentOutput.cols * (yindOut) + 3*(xindOut) + 1]=(uchar)(gOut +Scale*(1.0 - boolShiftGrain*.5)*gMed);
                    output[3*CurrentOutput.cols * (yindOut) + 3*(xindOut) + 2]=(uchar)(rOut + Scale*(1.0 - boolShiftGrain*.5)*rMed);
                    
                    
                
                
                } //y
            } //x
            
        }//m
        
        if ((fc%jp) == jp-1) {
            FirstGrains[g] = SecondGrains[g];
        }
        
        
        
    }//g
    
    }
    
}




//--------------------------------------------------------------
void testApp::draw(){
	ofSetHexColor(0xffffff);
    ofxCvColorImage AuxDrawImage;
 
    AuxDrawImage.allocate(TheFramesInput[fc].cols, TheFramesInput[fc].rows);
 
    AuxDrawImage = TheFramesInput[fc].data;
    AuxDrawImage.draw(0, 0);
    
    ofxCvColorImage AuxDrawImage2;
       
    AuxDrawImage2.allocate(CurrentOutput.cols, CurrentOutput.rows);
    AuxDrawImage2 = CurrentOutput.data;
    
    AuxDrawImage2.draw(0, AuxDrawImage.height);
   

    
    // Ploting the  mask
    glEnable(GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    
    for (int g =0; g < FirstGrains.size(); g++) {
        ofSetColor(0, 0, 0,100);
        ofNoFill();
        ofCircle(2*(FirstGrains[g].x +Gs/2.0) + 2*Nx, 2*(FirstGrains[g].y +Gs/2.0), 2*Gs/2.0);
        ofSetColor(60 + 160*(GrainMask[g].z), 0, 60,100);
        ofFill();
        ofCircle(2*(FirstGrains[g].x +Gs/2.0) + 2*Nx, 2*(FirstGrains[g].y +Gs/2.0), 2*Gs/2.0);
    }
    
    glDisable(GL_BLEND);

    fc++;
    if(fc>BlockDims.z-1){fc=0;}
    
}


void testApp::exit(){

  
 //   vidGrabber1.close();
   // vidGrabber2.close();


}

void InitGrains(){
// calculates the grains input and output positions.
    FirstGrains.clear();
    SecondGrains.clear();
    OutGrains.clear();
    
    // input jump
    int Osamp = (int)(ceil(Gs*inOp/100.0));
    int jp = Gs - Osamp;
    
    // output jump
    int outOsamp = (int)(ceil(Gs*outOp/100.0));
    int ojp = Gs - outOsamp;
    
    int graincounter = 0;
    
    for (int y = 0; y < (BlockDims.y - Gs); y+=jp) {
        for (int x=0; x< (BlockDims.x - Gs); x+=jp) {
                ofVec3f tempStorage;
                tempStorage.x = x;
                tempStorage.y=y;
                tempStorage.z=0;
                FirstGrains.push_back(tempStorage);
                SecondGrains.push_back(tempStorage);
                OutGrains.push_back(tempStorage);
                GrainMask.push_back(tempStorage);
                randAng.push_back(ofRandom(2*PI));
                GrainsRandom.push_back(graincounter);
            graincounter++;
        }
    }

    // shufling the random grain array:
    
    for (int k = graincounter-1 ; k >= 0; k--) {
        int indi = (int)ofRandom(k);
        int tempo = GrainsRandom[k];
        GrainsRandom[k] = GrainsRandom[indi];
        GrainsRandom[indi] = tempo;
        cout<<"k: "<<k<<", indi: "<< GrainsRandom[k]<<endl;
    }
    
    
    
}


void GrainRandomizer(){
    for (int g =0; g < SecondGrains.size(); g++) {
        
        if(GrainMask[g].z ==1.0){
            SecondGrains[g].z = (int)ofRandom(Nlast/4.0);
        }
    
    
    }
    
    
}


float CalculateScalecorrect(void){

    vector<float> Tempovec(10*Gs);
    // initializing
    for (int k=0; k<(10*Gs); k++) {
        Tempovec[k]=0.0;
    }
    
    // output jump
    int outOsamp = (int)(ceil(Gs*outOp/100.0));
    int ojp = Gs - outOsamp;
    
    // overlaping
    for (int k=0; k<(9*Gs); k+=ojp) {
        for (int n=0; n<Gs; n++) {
            Tempovec[k+n]+= 0.5*(1.0 -cosf(2*PI*n/(float)(Gs-1)));
        }
    }

    // finding the max;
    float TheMax =0;
    for (int k=0; k<(10*Gs); k++) {
        if(Tempovec[k]>TheMax){
            TheMax = Tempovec[k];
        }
    }

    return TheMax;

}

// Face detection function

/** @function detectAndDisplay */
void testApp::detectFace( cv::Mat frame )
{
    std::vector<cv::Rect> faces;
    cv::Mat frame_gray;
    
    cv::cvtColor( frame, frame_gray, CV_BGR2GRAY );
    cv::equalizeHist( frame_gray, frame_gray );
    
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(60, 60) );
    
    //-- Detect faces
    //    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, cv::Size(30, 30));
    
    //    cout<<faces.size()<<endl;
    for( int i = 0; i < faces.size(); i++ )
    {
         // Saving th face number 1 for recentering.
        
        if (i==0){
            FirstFace = faces[0];
        }
        
      }
    
}





// Motion detection function
// It marks as active the grains above a threshold
// The number of previous frames to be evaluated is an input parameter.


void testApp::detectMotion(int Nprev){
    
    
    // the current image:
    float ThMotion = 50;
    cv::Mat Graycurr;
    cv::cvtColor(TheFramesInput[fc], Graycurr, CV_RGB2GRAY);
    cv::Mat Acumm =cv::Mat::zeros(Graycurr.rows,Graycurr.cols, CV_8UC1);

    for (int k =0; k < Nprev; k++) {
        
        // reading a frame in the circular buffer:
        
        int LocalIndi = fc+ k +1;
        if(LocalIndi >= Nlast){LocalIndi = 0;}
        cv::Mat AuxGray;
        cv::cvtColor(TheFramesInput[LocalIndi], AuxGray, CV_RGB2GRAY);
        
        // calculating the absolute diferences with the next one
        cv::Mat TempoDiff;
        cv::absdiff(Graycurr, AuxGray, TempoDiff);
        // acumulating differences
        cv::add(Acumm, TempoDiff, Acumm);
        // updating the frame
        AuxGray.copyTo(Graycurr);
     }
    
    // Turning on grains depending on  changes:
    
    for (int g =0; g < FirstGrains.size(); g++) {
    
        unsigned char *inputAcumm = (unsigned char*)(Acumm.data);
        
        float GrainDiffSum = 0.0;
        int xindAcum,yindAcum;
        for (int x =0; x<Gs; x++) {
            for (int y=0; y<Gs; y++) {
                
                xindAcum = x + FirstGrains[g].x;
                yindAcum = y + FirstGrains[g].y;
                
                GrainDiffSum += (float)inputAcumm[Acumm.cols * (yindAcum) + (xindAcum) ];
            
            }
        }
    
        if (GrainDiffSum > Gs*Gs*ThMotion) {
            
             GrainMask[g].z = 1.0;
            
        }
        else{
        
             GrainMask[g].z = 0.0;
        }
    
    
    }
    
    
      UpdateVar = true;
    

}








//--------------------------------------------------------------
void testApp::keyPressed(int key){
    
    switch (key) {
        case 'u':


            break;
            
        case 'r':
            boolDelaygrains = !boolDelaygrains;

            
            break;
        case OF_KEY_RIGHT:
            NewoutOp++;

            
            break;
        case OF_KEY_LEFT:
            NewoutOp--;
            
            break;
        case 't':
 

            break;
            
        case OF_KEY_RETURN:
   
            boolDelaygrains = false;
            NewoutOp = inOp;

            break;
        case 'q':
            boolRotGrains =!boolRotGrains;
            break;
        default:
            break;
    }
    
    
}


//
void testApp::newMidiMessage(ofxMidiMessage& msg) {
    
	
	switch(msg.control){
            // slidders:
            case 0: // slidder 1 rotation
                SLD1 = msg .value;
            break;
            case 1: // slidder 2 time delay
                SLD2 = msg .value;
                UpdateVar =true;
            break;
            case 2: // slidder 3 hue
            SLD3 = msg .value;
            break;
            case 3: // slidder 4 sat
            SLD4 = msg .value;
            break;
            case 4: // slidder 4 sat
            SLD5 = msg .value;
            break;
            case 5:
            SLD6 = msg .value;
            break;
            case 6:
            SLD7 = msg .value;
            break;
            
            
            // Buttons:
            
            // Select none
            case 45:
            for (int g=0; g < GrainMask.size(); g++) {
                GrainMask[g].z = 0.0;
            }
            UpdateVar =true;
            
            
            break;
            
            // Select all
        case 41:
            for (int g=0; g < GrainMask.size(); g++) {
               GrainMask[g].z = 1.0;
            }
            UpdateVar =true;
            
            
            break;
            
        
            // stop cameras
            
        case 42:
            vidGrabber1.close();
            vidGrabber2.close();
            std::exit(1);
            
            break;
            
        //  select random
        case 44:
            
            for (int g=0; g < GrainMask.size(); g++) {
                GrainMask[g].z = (ofRandom(1.0)> 0.5);
            }
            UpdateVar =true;
            
            
        break;
            
            case 59:
                if(msg.value >100){
                    boolFixedDelay = !boolFixedDelay;
                    UpdateVar = true;
                }
            break;
            case 64:
            boolRotGrains = true;
            break;
            case 48:
            boolRotGrains = false;
            break;
            case 65:
            if(msg.value >100){
                boolDelaygrains = true;
                UpdateVar =true;
            }
            break;
            
        case 50:
            boolhueshift = false;
            break;
        case 66:
            if(msg.value >100){
                boolhueshift = true;
            }
            break;
        case 51:
            boolSatChange = false;
            break;
        case 67:
            if(msg.value >100){
                boolSatChange = true;
            }
            break;
        case 53:
            boolShiftGrain = false;
            break;
        case 69:
            if(msg.value >100){
                boolShiftGrain = true;
            }
            break;
            
            
        case 60:
            if(msg.value >100){
                boolCam2Grain = !boolCam2Grain;
            }
            break;

            
        case 61:
            if(msg.value >100){
                boolChangeDet = !boolChangeDet;
            }
            break;
            
        case 52:
            boolGeomGrains = false;
            break;
        case 68:
            if(msg.value >100){
                boolGeomGrains = true;
            }
            break;
            
        case 46:
            if(msg.value >100){
                boolFace = !boolFace;
            }
            break;
            
            case 49:
            if(msg.value >100){
                if(boolDelaygrains){
                    // reset delays
                    for (int g =0; g < SecondGrains.size(); g++) {
                        SecondGrains[g].z = 0.0;
                    }
                }
                boolDelaygrains = false;
            }
            break;
    }
    
}

// Audio callback

void testApp::audioIn(float *Ainput, int BufferSize, int nChannels){

// samples are interleaved
    float SampleAverage =0;
    float RMSValue = 0;
    for (int i = 0; i < BuffS; i++) {
       // SampleAverage = Ainput[i*2]*0.5 + Ainput[i*2+1]*0.5;
        SampleAverage = Ainput[i];
        AudioValues[i] = SampleAverage;
        RMSValue += SampleAverage*SampleAverage;
    }
    CurrentEnergy = RMSValue/BuffS;
}



// Audio based selections

void testApp::EnergyCenterSelect(){

    
    float centerX = BlockDims.x/2.0;
    float centerY = BlockDims.y/2.0;
    //cout<<"Cx: "<<centerX<<", Cy: "<<centerY<<endl;
    float ThAudio = 0.05;
    for (int g=0; g < GrainMask.size(); g++) {
    
        float Xpos = GrainMask[g].x+Gs/2.0;
        float Ypos = GrainMask[g].y+Gs/2.0;
        
        float FarMeasure = 1.0 - 2*sqrtf((Xpos-centerX)*(Xpos-centerX)
                                         +(Ypos-centerY)*(Ypos-centerY))/sqrtf(Nx*Nx+Ny*Ny);
        
       //cout <<FarMeasure*CurrentEnergy<<endl;
        if (SLD7/127.0*FarMeasure*sqrtf(CurrentEnergy)> ThAudio){
            GrainMask[g].z = 1.0;
          }
        else{
            GrainMask[g].z = 0.0;
        }
    
    }

    UpdateVar = true;

}

void testApp::AudioRandomSelect(){
    

    for (int g=0; g < GrainMask.size(); g++) {
        
  
        if (g < 50*SLD7/127.0*GrainMask.size()*sqrtf(CurrentEnergy)){
            GrainMask[GrainsRandom[g]].z = 1.0;
        }
        else{
            GrainMask[GrainsRandom[g]].z = 0.0;
        }
        
    }
    
    UpdateVar = true;
    
}





//--------------------------------------------------------------
void testApp::keyReleased(int key){
    
}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y){
    
}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){
    
}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){
    
    for (int g=0; g < GrainMask.size(); g++) {
        
        float distX = 2*(GrainMask[g].x +Gs/2.0) + 2*Nx - x;
        float distY = 2*(GrainMask[g].y +Gs/2.0)-y;
        if((distX*distX + distY*distY) < 30*30){
            GrainMask[g].z = (GrainMask[g].z==0);
        }
        
        
    }
    UpdateVar =true;

}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){
    
}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){
    
}

//--------------------------------------------------------------
void testApp::gotMessage(ofMessage msg){
    
}

//--------------------------------------------------------------
void testApp::dragEvent(ofDragInfo dragInfo){ 
    
}