import React, { useEffect, useState } from "react";
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";

export default function Analysis(props) {
  //gets the props and gets the information needed from them
  const url = props.post.url;
  const post = props.post;
  const setPost = props.setCurrPost;
  const caption = props.post.caption;
  const sentiment = props.post.sentiment;
  const proba = props.post.proba;
  const topic = props.post.topic;
  const type = props.post.type;

  //implements back button so the user can get back from the post analysis to the profile page
  function BackButtonFunc(props) {
    const setPost = props.setPost;
    return (
      <button
        className="backButton"
        title="Go Back"
        onClick = {()=>setPost(null)}
      >Go Back</button>
    );
  }

  const value = (proba * 100).toFixed(0); //sets post sentiment percentage to be a whole number percentage

  //return shows all the features seen in the post analysis page of the application 
  return (
    <div className="analysisContainer">
      <div className="imgContainer">
        {Array.isArray(url) ? (
          url.map((arrUrl) => <img src={arrUrl}></img>)
        ) : type == "VIDEO" ? (
          <video className="selectedVid" controls>
            {" "}
            <source src={url} type="video/mp4" />{" "}
          </video>
        ) : (
          <img className="selectedImg" src={url}></img>
        )}{" "}
      </div>
      <div className="captionContainer">
        <><BackButtonFunc setPost={setPost}/></>
        <p className="caption">{caption}</p>
        <p className="sentiment">Sentiment Analysis:</p>
        <div className="circleContainer">
          <CircularProgressbar
            value={value}
            maxValue={100}
            text={`${value}% ${sentiment == 1 ? "Positive" : (sentiment == 0 ? "Negative" : "Neutral") }`}
            styles={buildStyles({
              pathColor: `#BC3DF8`,
              trailColor: `#D3D3D3`,
              textColor: `#A020F0`,
              textSize: 12,
            })}
          />
        </div>
        <p className="topic">Topic Analysis: {topic}</p>
      </div>
    </div>
  );
}
