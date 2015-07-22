/**
 * Copyright 2015 Brigham Young University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.byu.nlp.crowdsourcing;

import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Type;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.random.RandomGenerator;

import com.google.common.base.Charsets;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.reflect.TypeToken;
import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;

import edu.byu.nlp.classify.eval.Prediction;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.crowdsourcing.models.em.CSLDADiscreteModel;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.io.Files2;
import edu.byu.nlp.util.Matrices;

/**
 * @author plf1
 *
 * Contains the minimal state necessary to save a preserve 
 * the location of a crowdsourcing model, indexed by 
 * document source (so that it can be used to initialize 
 * a new model that has the data in a different order).  
 * 
 */
public class SerializableCrowdsourcingState  {

  
  public static class SerializableCrowdsourcingDocumentState{

    private Integer y;
    private int[] z;
    private Integer m;

    public static SerializableCrowdsourcingDocumentState of(Integer y, Integer m, int[] z) {
        SerializableCrowdsourcingDocumentState state = new SerializableCrowdsourcingDocumentState();
        state.y = y;
        state.m = m;
        state.z = z;
        return state;
    }
    public static SerializableCrowdsourcingDocumentState of(int[] y, int[] m, int[][] z, int i) {
    	return of(
    			(y==null)? null: y[i], 
				(m==null)? null: m[i],
				(z==null)? null: z[i]
				);
    }
    
    public Integer getY(){
      return y;
    }
    public Integer getM(){
      return m;
    }
    public int[] getZ(){
      return z;
    }
  }
  
  
  private Map<String,SerializableCrowdsourcingDocumentState> perDocumentState;
  private double goodness; // some transient (non-serialized) measure of this model state's goodness
  private boolean hasY;
  private boolean hasM;
  private boolean hasZ;
  
  
  Map<String,SerializableCrowdsourcingDocumentState> getPerDocumentState(){
    return perDocumentState;
  }
  
  public void serializeTo(String filename) throws IOException{
    Files2.write(new Gson().toJson(perDocumentState), filename);
  }

  public double getGoodness() {
    return goodness;
  }

  public void setGoodness(double goodness) {
    this.goodness = goodness;
  }
  
  public SerializableCrowdsourcingDocumentState getDocument(String docSource){
	  return perDocumentState.get(docSource);
  }

  public void serializeTo(PrintWriter serializeOut) {
	 serializeOut.write(new Gson().toJson(perDocumentState));
  }

  public boolean hasY(){
	  return hasY;
  }

  public boolean hasM(){
	  return hasM;
  }

  public boolean hasZ(){
	  return hasZ;
  }
  
  
  

  public static SerializableCrowdsourcingState of(Map<String, SerializableCrowdsourcingDocumentState> perDocumentState) {
    SerializableCrowdsourcingState model = new SerializableCrowdsourcingState();
    model.perDocumentState = perDocumentState;
    // trust that the first document is representative of the rest. If it has a Y, all
    // docs must have Y, etc.
    SerializableCrowdsourcingDocumentState firstDoc = perDocumentState.values().iterator().next();
    model.hasY = firstDoc.getY()!=null;
    model.hasM = firstDoc.getM()!=null;
    model.hasZ = firstDoc.getZ()!=null;
    return model;
  }
  
  public static SerializableCrowdsourcingState of(MultiAnnModel model){
    int[] y = model.getCurrentState().getY();
    int[] m = model.getCurrentState().getM();
    String[] docSources = Datasets.docRawSourcesIn(model.getCurrentState().getData());
    int[][] z = null;
    return of(docSources, y, m, z);
  }

  public static SerializableCrowdsourcingState of(Predictions predictions){
	  Map<String, SerializableCrowdsourcingDocumentState> perDocumentState = Maps.newHashMap();
	  for (Prediction pred: predictions.allPredictions()){
		  String src = pred.getInstance().getInfo().getRawSource();
		  Integer y = pred.getPredictedLabel();
		  Integer m = pred.getAlternativePredictedLabel();
		  int[] z = pred.getPredictedWordTopics();
		  perDocumentState.put(src, SerializableCrowdsourcingDocumentState.of(y, m, z));
	  }
	  return of(perDocumentState);
  }
  
  public static SerializableCrowdsourcingState of(CSLDADiscreteModel model){
    int[] y = model.getY();
    int[] m = null;
    int[][] z = model.getZ();
    String[] docSources = Datasets.docRawSourcesIn(model.getData());
    return of(docSources, y, m, z);
  }

  public static SerializableCrowdsourcingState of(String[] docSources, int[] y, int[] m, int[][] z){
    Preconditions.checkNotNull(docSources, "docSources are required");
    Preconditions.checkArgument(docSources.length>0, "at least one document is required!");

    // ensure all non-null vectors are the same length
    Preconditions.checkArgument(y==null || y.length==docSources.length, "vector y must either be null or the same length as docSources");
    Preconditions.checkArgument(m==null || m.length==docSources.length, "vector m must either be null or the same length as docSources");
    Preconditions.checkArgument(z==null || z.length==docSources.length, "vector z must either be null or the same length as docSources");
    
    
    Map<String,SerializableCrowdsourcingDocumentState> perDocumentState = Maps.newHashMap();
    for (int i=0; i<docSources.length; i++){
      perDocumentState.put(docSources[i], SerializableCrowdsourcingDocumentState.of(y,m,z,i));
    }
    return SerializableCrowdsourcingState.of(perDocumentState);
  }

  
  public static SerializableCrowdsourcingState deserializeFrom(String filename) throws JsonSyntaxException, IOException{
    @SuppressWarnings("serial")
    Type type = new TypeToken<Map<String,SerializableCrowdsourcingDocumentState>>(){}.getType();
    Map<String,SerializableCrowdsourcingDocumentState> perDocumentState = new Gson().fromJson(Files2.toString(filename, Charsets.UTF_8), type);
    return SerializableCrowdsourcingState.of(perDocumentState);
  }

  public static int[] orderedY(SerializableCrowdsourcingState state, String[] docSourceOrder){
	int[] assignments = new int[docSourceOrder.length];
	for (int i=0; i<docSourceOrder.length; i++){
		SerializableCrowdsourcingDocumentState docState = state.getDocument(docSourceOrder[i]);
		assignments[i] = docState.getY(); 
	}
	return assignments;
  }

  public static int[] orderedM(SerializableCrowdsourcingState state, String[] docSourceOrder){
		int[] assignments = new int[docSourceOrder.length];
		for (int i=0; i<docSourceOrder.length; i++){
			SerializableCrowdsourcingDocumentState docState = state.getDocument(docSourceOrder[i]);
			assignments[i] = docState.getM(); 
		}
		return assignments;
  }

  public static int[][] orderedZ(SerializableCrowdsourcingState state, String[] docSourceOrder){
	int[][] assignments = new int[docSourceOrder.length][];
	for (int i=0; i<docSourceOrder.length; i++){
		SerializableCrowdsourcingDocumentState docState = state.getDocument(docSourceOrder[i]);
		assignments[i] = docState.getZ(); 
	}
	return assignments;
  }

  public static int[][] yChains(List<SerializableCrowdsourcingState> chains, String[] docSourceOrder){
	  List<int[]> yChains = Lists.newArrayList();
	  for (SerializableCrowdsourcingState chain: chains){
		  yChains.add(orderedY(chain, docSourceOrder));
	  }
	  return yChains.toArray(new int[][]{});
  }

  public static int[][] mChains(List<SerializableCrowdsourcingState> chains, String[] docSourceOrder){
	  List<int[]> mChains = Lists.newArrayList();
	  for (SerializableCrowdsourcingState chain: chains){
		  mChains.add(orderedM(chain, docSourceOrder));
	  }
	  return mChains.toArray(new int[][]{});
  }

  public static int[][][] zChains(List<SerializableCrowdsourcingState> chains, String[] docSourceOrder){
	  List<int[][]> mChains = Lists.newArrayList();
	  for (SerializableCrowdsourcingState chain: chains){
		  mChains.add(orderedZ(chain, docSourceOrder));
	  }
	  return mChains.toArray(new int[][][]{});
  }

  public static boolean chainsHaveY(List<SerializableCrowdsourcingState> chains){
	  return chains!=null && chains.size()>0 && chains.get(0).hasY();
  }

  public static boolean chainsHaveM(List<SerializableCrowdsourcingState> chains){
	  return chains!=null && chains.size()>0 && chains.get(0).hasM();
  }
  
  public static boolean chainsHaveZ(List<SerializableCrowdsourcingState> chains){
	  return chains!=null && chains.size()>0 && chains.get(0).hasZ();
  }

  public static SerializableCrowdsourcingState majorityVote(List<SerializableCrowdsourcingState> initializationChains, RandomGenerator rnd) {
	  if (initializationChains==null){
		  return null;
	  }
	  Preconditions.checkArgument(initializationChains.size()>0);
	  String[] arbitraryDocOrder = initializationChains.get(0).getPerDocumentState().keySet().toArray(new String[]{});

	  int[][] yChains = yChains(initializationChains, arbitraryDocOrder);
	  int[] y = Matrices.aggregateRowsViaMajorityVote(yChains, rnd);
	  
	  int[][] mChains = mChains(initializationChains, arbitraryDocOrder);
	  int[] m = Matrices.aggregateRowsViaMajorityVote(mChains, rnd);

	  int[][][] zChains = zChains(initializationChains, arbitraryDocOrder);
	  int[][] z = Matrices.aggregateFirstDimensionViaMajorityVote(zChains, rnd);
	  
	  return of(arbitraryDocOrder, y, m, z);
  }
  
}
