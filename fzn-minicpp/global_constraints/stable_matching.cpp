//Based on the paper:
//  "An n-ary Constraint for the Stable Marriage Problem"
//  by Chris Unsworth and Patrick Prosser
//  (http://arxiv.org/abs/1308.0183)


#include "stable_matching.hpp"

StableMatching::StableMatching(std::vector<var<int>::Ptr> & m, std::vector<var<int>::Ptr> & w, std::vector<std::vector<int>> const & mpl, std::vector<std::vector<int>> const & wpl) :
    Constraint(m[0]->getSolver()), _x(m), _y(w), _xpl(mpl), _ypl(wpl)
{
    setPriority(CLOW);

    // Get the size of the problem instance
    _n = static_cast<int>(_x.size());

    // Build inverse matrices
    _xPy = (int *)malloc(_n * _n * sizeof(int));
    _yPx = (int *)malloc(_n * _n * sizeof(int));
    buildReverseMatrix(_xpl,_xPy);
    buildReverseMatrix(_ypl,_yPx);

    //Initialize ylb, yub, xlb and xub
    for (int i = 0; i < 10; i  += 1)
    {
        _yub.push_back(trail<int>(m[0]->getSolver()->getStateManager(), _n-1));
        _ylb.push_back(trail<int>(m[0]->getSolver()->getStateManager(), 0));
        _xub.push_back(trail<int>(m[0]->getSolver()->getStateManager(), _n-1));
        _xlb.push_back(trail<int>(m[0]->getSolver()->getStateManager(), 0));
    }
}

void StableMatching::post(){
    for (auto const & v : _x)
    {
        v->propagateOnDomainChange(this);
    }

    for (auto const & v : _y)
    {
        v->propagateOnDomainChange(this);
    }
    init();

    propagateOnQueue();
}

void StableMatching::propagate(){
    //Fills queue
    for(int i=0; i<_n; i++){
        if(_x[i]->min()!=_xlb[i]){ //Check if min changed using xlb (more precise than changedMin())
            _callQueue.push(constraintCall(1,i,0,0));
        }
        if(_y[i]->max()!=_yub[i]){ //Check if max changed using yub (more precise than changedMin())
            _callQueue.push(constraintCall(2,i,0,0));
        }
        if(_x[i]->changed()){ //Search domain for removed values only if variable is marked as changed
            for(int k=_x[i]->min()+1;k<=_xub[i];k++){
                if(!_x[i]->contains(k)){
                    _callQueue.push(constraintCall(0,i,k,1));
                }
            }
        }
        //Applies remove value on the women too (this is missing from the original paper)
        if(_y[i]->changed()){//Search domain for removed values only if variable is marked as changed
            for(int k=_ylb[i];k<_y[i]->max();k++){
                if(!_y[i]->contains(k)){
                    _callQueue.push(constraintCall(0,i,k,0));
                }
            }
        }
    }

    propagateOnQueue();
}

void StableMatching::buildReverseMatrix(std::vector<std::vector<int>> zpl, int *zPz){
    for(int i=0;i<_n;i++){
        for(int j=0;j<_n;j++){
            zPz[i*_n+zpl[i][j]]=j;
        }
    }
}

void StableMatching::propagateOnQueue(){
    // Executes functions until the queue is empty
    while(!_callQueue.empty()){
        functionDispatcher();
    }

    //Updates bounds not used by the subfunctions
    for(int i=0;i<_n;i++){
        _ylb[i]=_y[i]->min();
        _xub[i]=_x[i]->max();
    }
}

void StableMatching::functionDispatcher(){
    constraintCall c = _callQueue.front();
    switch (c.function) {
        case 0: // removeValue
            removeValue(c.ij,c.a,c.isMan);
            break;

        case 1: // deltaMin
            deltaMin(c.ij);
            break;

        case 2: // deltaMax
            deltaMax(c.ij);
            break;
    }
    _callQueue.pop();
}

void StableMatching::removeValue(int i, int a, int isMan){
    if (isMan){
        int j = _xpl[i][a];
        if (_y[j]->contains(_yPx[j*_n+i])){
            if(_y[j]->max()==_yPx[j*_n+i]){
                //Adds deltaMax to queue
                _callQueue.push(constraintCall(2,j,0,0));
            }
            _y[j]->remove(_yPx[j*_n+i]);
        }
    } else {
        int j = _ypl[i][a];
        if(_x[j]->contains(_xPy[j*_n+i])){
            if(_x[j]->min()==_xPy[j*_n+i]){
                //Adds deltaMin to queue
                _callQueue.push(constraintCall(1,j,0,0));
            }
            _x[j]->remove(_xPy[j*_n+i]);
        }
    }
}

void StableMatching::deltaMin(int i){  //Note: we know that this won't ever be launched on an empty domain
    int j = _xpl[i][_x[i]->min()];
    if(_y[j]->max()>_yPx[j*_n+i]){
        _y[j]->removeAbove(_yPx[j*_n+i]);
        //Adds deltaMax to queue
        _callQueue.push(constraintCall(2,j,0,0));
    }
    for(int k=_xlb[i]; k<_x[i]->min();k++){
        j = _xpl[i][k];
        if (_y[j]->max()>_yPx[j*_n+i]-1){
            _y[j]->removeAbove(_yPx[j*_n+i]-1);
            //Adds deltaMax to queue
            _callQueue.push(constraintCall(2,j,0,0));
        }
    }
    _xlb[i] = _x[i]->min();
}

void StableMatching::deltaMax(int j){
    int i;
    for(int k=_y[j]->max()+1;k<=_yub[j];k++){
        i = _ypl[j][k];
        if(_x[i]->contains(_xPy[i*_n+j])){
            if(_x[i]->min()==_xPy[i*_n+j]){
                _x[i]->remove(_xPy[i*_n+j]);
                //Adds deltaMin to queue
                _callQueue.push(constraintCall(1,i,0,0));
            } else {
                _x[i]->remove(_xPy[i*_n+j]);
                //Adds removeValue to queue
                _callQueue.push(constraintCall(0,i,_xPy[i*_n+j],1));
            }
        }
    }
    _yub[j]=_y[j]->max();
}

void StableMatching::init(){
    for(int i=0;i<_n;i++){
        deltaMin(i);
    }
}
