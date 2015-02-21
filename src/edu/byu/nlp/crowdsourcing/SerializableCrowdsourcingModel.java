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
import java.lang.reflect.Type;
import java.util.Map;

import com.google.common.base.Charsets;
import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.google.common.reflect.TypeToken;
import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;

import edu.byu.nlp.crowdsourcing.em.ConfusedSLDADiscreteModel;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.io.Files2;

/**
 * @author plf1
 *
 * Contains the minimal state necessary to save a preserve 
 * the location of a crowdsourcing model, indexed by 
 * document source (so that it can be used to initialize 
 * a new model that has the data in a different order).  
 * 
 */
public class SerializableCrowdsourcingModel  {

  
  public static class SerializableCrowdsourcingModelDocument{

    private Integer y;
    private int[] z;
    private Integer m;
    
    public static SerializableCrowdsourcingModelDocument of(int[] y, int[] m, int[][] z, int i) {
      SerializableCrowdsourcingModelDocument state = new SerializableCrowdsourcingModelDocument();
      state.y = (y==null)? null: y[i];
      state.m = (m==null)? null: m[i];
      state.z = (z==null)? null: z[i];
      return state;
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
  
  
  private Map<String,SerializableCrowdsourcingModelDocument> perDocumentState;
  private double goodness; // some transient (non-serialized) measure of this model state's goodness
  
  
  Map<String,SerializableCrowdsourcingModelDocument> getPerDocumentState(){
    return perDocumentState;
  }
  
  public static SerializableCrowdsourcingModel of(Map<String, SerializableCrowdsourcingModelDocument> perDocumentState) {
    SerializableCrowdsourcingModel model = new SerializableCrowdsourcingModel();
    model.perDocumentState = perDocumentState;
    return model;
  }
  
  public static SerializableCrowdsourcingModel of(MultiAnnModel model){
    int[] y = model.getCurrentState().getY();
    int[] m = model.getCurrentState().getM();
    String[] docSources = Datasets.docSourcesIn(model.getCurrentState().getData());
    int[][] z = null;
    return of(docSources, y, m, z);
  }

  public static SerializableCrowdsourcingModel of(ConfusedSLDADiscreteModel model){
    int[] y = model.getY();
    int[] m = null;
    int[][] z = model.getZ();
    String[] docSources = Datasets.docSourcesIn(model.getData());
    return of(docSources, y, m, z);
  }

  public static SerializableCrowdsourcingModel of(String[] docSources, int[] y, int[] m, int[][] z){
    Preconditions.checkNotNull(docSources, "docSources are required");
    Preconditions.checkArgument(docSources.length>0, "at least one document is required!");

    // ensure all non-null vectors are the same length
    Preconditions.checkArgument(y==null || y.length==docSources.length, "vector y must either be null or the same length as docSources");
    Preconditions.checkArgument(m==null || m.length==docSources.length, "vector m must either be null or the same length as docSources");
    Preconditions.checkArgument(z==null || z.length==docSources.length, "vector z must either be null or the same length as docSources");
    
    
    Map<String,SerializableCrowdsourcingModelDocument> perDocumentState = Maps.newHashMap();
    for (int i=0; i<docSources.length; i++){
      perDocumentState.put(docSources[i], SerializableCrowdsourcingModelDocument.of(y,m,z,i));
    }
    return SerializableCrowdsourcingModel.of(perDocumentState);
  }

  
  public static SerializableCrowdsourcingModel deserializeFrom(String filename) throws JsonSyntaxException, IOException{
    @SuppressWarnings("serial")
    Type type = new TypeToken<Map<String,SerializableCrowdsourcingModelDocument>>(){}.getType();
    Map<String,SerializableCrowdsourcingModelDocument> perDocumentState = new Gson().fromJson(Files2.toString(filename, Charsets.UTF_8), type);
    return SerializableCrowdsourcingModel.of(perDocumentState);
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
  
}
